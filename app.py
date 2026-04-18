# =============================================================================
# app.py — Gradio Demo: Speech Emotion Recognition
#
# Runs locally:     python app.py
# HuggingFace Space: upload this file + wav2vec2_ser.py + config.py +
#                    best_model.pth + requirements_space.txt to your Space
#
# Input:  .wav file (any sample rate — resampled internally)
# Output: predicted emotion + confidence bar chart
# =============================================================================

import os
import sys

import numpy as np
import torch
import gradio as gr
import librosa

sys.path.insert(0, os.path.dirname(__file__))
import config
from models.wav2vec2_ser import Wav2Vec2SER

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

EMOTIONS    = config.EMOTIONS          # {0:"angry", 1:"fear", 2:"happy", 3:"neutral", 4:"sad"}
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# On HuggingFace Spaces, best_model.pth sits at the Space root.
# Locally it lives inside outputs/model9_wav2vec2/.
_hf_ckpt    = os.path.join(os.path.dirname(__file__), "best_model.pth")
_local_ckpt = os.path.join(config.OUTPUTS_DIR, "model9_wav2vec2", "best_model.pth")
CKPT_PATH   = _hf_ckpt if os.path.isfile(_hf_ckpt) else _local_ckpt

# Emotion display labels and colors for the bar chart
EMOTION_COLORS = {
    "angry":   "#e74c3c",
    "fear":    "#9b59b6",
    "happy":   "#f1c40f",
    "neutral": "#95a5a6",
    "sad":     "#3498db",
}

# ---------------------------------------------------------------------------
# LOAD MODEL (once at startup)
# ---------------------------------------------------------------------------

def _load_model():
    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CKPT_PATH}\n"
            "Run training first: py run_pipeline.py --step train9"
        )
    model = Wav2Vec2SER()
    state = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded from {CKPT_PATH} → {DEVICE}")
    return model

MODEL = _load_model()

# ---------------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------------

def preprocess_audio(audio_path: str) -> torch.Tensor:
    """
    Load any audio file and convert it to a (1, 88200) waveform tensor
    that matches exactly what the model was trained on.

    Steps:
      1. Load at 22050 Hz (mono) — same as training
      2. Center-pad or center-crop to exactly 4 seconds (88200 samples)
      3. Return as float32 tensor with batch dimension
    """
    y, _ = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)

    target = config.N_SAMPLES  # 88200

    if len(y) >= target:
        # Center-crop: take the middle 4 seconds
        start = (len(y) - target) // 2
        y = y[start: start + target]
    else:
        # Center-pad: add silence symmetrically on both sides
        pad_total = target - len(y)
        pad_left  = pad_total // 2
        pad_right = pad_total - pad_left
        y = np.pad(y, (pad_left, pad_right), mode="constant")

    waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0)  # (1, 88200)
    return waveform

# ---------------------------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------------------------

def predict_emotion(audio_path: str):
    """
    Full inference pipeline: audio file → emotion prediction + confidence chart.

    Returns:
        label_text: str  — primary prediction with confidence
        chart_data: dict — Gradio BarPlot-compatible dict
    """
    if audio_path is None:
        return "Please upload or record an audio clip.", None

    try:
        waveform = preprocess_audio(audio_path)
    except Exception as e:
        return f"Error loading audio: {e}", None

    # Build the batch dict the model expects
    batch = {"waveform": waveform}

    with torch.no_grad():
        logits = MODEL(batch, DEVICE)                         # (1, 5)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (5,)

    # Build results
    pred_idx  = int(probs.argmax())
    pred_name = EMOTIONS[pred_idx]
    confidence = float(probs[pred_idx])

    # Primary label
    if confidence < config.CONFIDENCE_THRESHOLD:
        label_text = (
            f"Uncertain ({confidence:.0%} confidence)\n"
            "Try a longer or clearer recording."
        )
    else:
        label_text = f"{pred_name.upper()}  —  {confidence:.0%} confidence"

    # Bar chart data: list of (emotion, probability%) for Gradio
    bar_data = {
        "Emotion": [EMOTIONS[i].capitalize() for i in range(config.NUM_CLASSES)],
        "Confidence (%)": [round(float(p) * 100, 1) for p in probs],
    }

    return label_text, bar_data

# ---------------------------------------------------------------------------
# GRADIO INTERFACE
# ---------------------------------------------------------------------------

def run_interface(audio):
    """Wrapper that Gradio calls with the audio file path."""
    label, bar_data = predict_emotion(audio)

    if bar_data is None:
        return label, None

    import pandas as pd
    df = pd.DataFrame(bar_data)
    return label, df


with gr.Blocks(title="Speech Emotion Recognition", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # Speech Emotion Recognition
        **Upload a `.wav` file or record directly** — the model predicts one of five emotions:
        Angry · Fear · Happy · Neutral · Sad

        > Powered by **Wav2Vec2-base** fine-tuned on RAVDESS + CREMA-D (~7,000 clips, 115 speakers).
        > Test accuracy: **75.3%** on a speaker-independent held-out test set.
        """
    )

    with gr.Row():
        audio_input = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="Upload or record audio (.wav required)",
        )

    predict_btn = gr.Button("Predict Emotion", variant="primary")

    with gr.Row():
        label_out = gr.Textbox(
            label="Prediction",
            lines=2,
            interactive=False,
        )

    bar_out = gr.BarPlot(
        x="Emotion",
        y="Confidence (%)",
        title="Confidence per Emotion",
        y_lim=[0, 100],
        color="Emotion",
        label="Confidence breakdown",
    )

    predict_btn.click(
        fn=run_interface,
        inputs=[audio_input],
        outputs=[label_out, bar_out],
    )

    gr.Markdown(
        """
        ---
        **Notes:**
        - Speak clearly for at least 2–3 seconds
        - Model was trained on acted speech — performance on natural conversation may vary
        - Prediction shown as "Uncertain" when confidence < 50%
        - Best results with clean, single-speaker audio
        """
    )

if __name__ == "__main__":
    demo.launch(share=False)   # set share=True to get a temporary public link
