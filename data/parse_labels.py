# =============================================================================
# parse_labels.py
#
# Converts a raw audio file path into (actor_id, label, emotion_name, dataset).
# Returns None if the file should be skipped (e.g., disgust, calm).
#
# DESIGN RULE: All label mappings come from config.py — no hardcoded labels here.
# =============================================================================

import os
import sys

# Add Scripts root to path so we can import config from anywhere
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def parse_ravdess(filepath: str):
    """
    Parse a RAVDESS .wav filepath.

    RAVDESS filename format:
        [Modality]-[VocalChannel]-[EmotionCode]-[Intensity]-[Statement]-[Rep]-[Actor].wav
        Example: 03-01-05-01-01-01-07.wav
            → emotion code = "05" (angry), actor = "07"

    Returns:
        (actor_id, label, emotion_name, dataset) or None if the file should be skipped.
    """
    filename = os.path.basename(filepath)
    stem = os.path.splitext(filename)[0]  # remove .wav
    parts = stem.split("-")

    if len(parts) != 7:
        # Unexpected filename format — skip silently
        return None

    emotion_code = parts[2]   # position 2 (0-based) = emotion code
    actor_str    = parts[6]   # position 6 = actor number (e.g., "07")

    label = config.RAVDESS_EMOTION_MAP.get(emotion_code, None)
    if label is None:
        # This emotion code is not in our 5-class set (calm/disgust/surprised) — skip
        return None

    emotion_name = config.EMOTIONS[label]
    actor_id     = f"ravdess_{actor_str}"  # e.g., "ravdess_07"

    return actor_id, label, emotion_name, "RAVDESS"


def parse_cremad(filepath: str):
    """
    Parse a CREMA-D .wav filepath.

    CREMA-D filename format:
        [ActorID]_[Sentence]_[EmotionCode]_[Intensity].wav
        Example: 1001_DFA_ANG_XX.wav
            → emotion code = "ANG" (angry), actor = "1001"

    Returns:
        (actor_id, label, emotion_name, dataset) or None if the file should be skipped.
    """
    filename = os.path.basename(filepath)
    stem = os.path.splitext(filename)[0]
    parts = stem.split("_")

    if len(parts) < 3:
        return None

    actor_str    = parts[0]   # e.g., "1001"
    emotion_code = parts[2]   # e.g., "ANG"

    label = config.CREMAD_EMOTION_MAP.get(emotion_code, None)
    if label is None:
        # DIS (disgust) or unknown code — skip
        return None

    emotion_name = config.EMOTIONS[label]
    actor_id     = f"cremad_{actor_str}"  # e.g., "cremad_1001"

    return actor_id, label, emotion_name, "CREMA-D"


def parse_file(filepath: str):
    """
    Auto-detect dataset from the filepath and parse labels accordingly.

    Returns:
        (actor_id, label, emotion_name, dataset) or None if skipped.
    """
    fp_lower = filepath.lower().replace("\\", "/")

    if "ravdess" in fp_lower:
        return parse_ravdess(filepath)
    elif "crema-d" in fp_lower or "crema_d" in fp_lower:
        return parse_cremad(filepath)
    else:
        raise ValueError(
            f"Cannot determine dataset from path: {filepath}\n"
            f"Expected path to contain 'ravdess' or 'crema-d'."
        )


# ---------------------------------------------------------------------------
# Quick self-test (run: python data/parse_labels.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_cases = [
        # RAVDESS: emotion 05=angry, actor 07 → label 0
        (r"..\..\Data\RAVDESS\Actor_07\03-01-05-01-01-01-07.wav", (0, "angry", "ravdess_07", "RAVDESS")),
        # RAVDESS: emotion 02=calm → should be skipped (None)
        (r"..\..\Data\RAVDESS\Actor_01\03-01-02-01-01-01-01.wav", None),
        # CREMA-D: ANG=angry, actor 1001 → label 0
        (r"..\..\Data\CREMA-D\1001_DFA_ANG_XX.wav", (0, "angry", "cremad_1001", "CREMA-D")),
        # CREMA-D: DIS=disgust → should be skipped (None)
        (r"..\..\Data\CREMA-D\1001_DFA_DIS_XX.wav", None),
    ]

    all_passed = True
    for path, expected in test_cases:
        result = parse_file(path)
        if result is None:
            got = None
        else:
            actor_id, label, emotion_name, dataset = result
            got = (label, emotion_name, actor_id, dataset)

        passed = (got == expected)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] {os.path.basename(path)}")
        if not passed:
            print(f"       expected: {expected}")
            print(f"       got:      {got}")

    print()
    print("All tests passed!" if all_passed else "SOME TESTS FAILED.")
