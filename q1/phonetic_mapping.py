import os
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "q1_audio"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "q1" / "phonetic_mapping"


def find_first_wav(data_dir):
    wav_files = sorted(data_dir.glob("*.wav"))
    if not wav_files:
        return None
    return wav_files[0]


def load_audio(file_path, target_sr=16000):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    audio, sr = sf.read(str(file_path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    if sr != target_sr:
        raise ValueError(
            f"Audio sample rate is {sr}, but this script expects {target_sr} Hz.\n"
            f"Please use a 16 kHz WAV file."
        )

    return audio, sr


def get_dummy_manual_boundaries(audio, sr, step_sec=0.2):
    duration = len(audio) / sr
    return np.arange(0, duration, step_sec)


def get_model_boundaries(audio, sr):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    time_steps = logits.shape[1]
    duration = len(audio) / sr
    times = np.linspace(0, duration, time_steps)

    token_changes = []
    pred_seq = predicted_ids[0].cpu().numpy()

    for i in range(1, len(pred_seq)):
        if pred_seq[i] != pred_seq[i - 1]:
            token_changes.append(times[i])

    return transcription, np.array(token_changes)


def rmse_boundaries(manual_b, model_b, k=10):
    if len(manual_b) == 0 or len(model_b) == 0:
        return None

    n = min(len(manual_b), len(model_b), k)
    return np.sqrt(np.mean((manual_b[:n] - model_b[:n]) ** 2))


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_file = find_first_wav(DATA_DIR)

    if input_file is None:
        raise FileNotFoundError(
            f"No .wav file found in {DATA_DIR}\n"
            f"Please place at least one 16 kHz speech WAV file there."
        )

    print(f"Using audio file: {input_file}")

    audio, sr = load_audio(input_file, 16000)

    manual_boundaries = get_dummy_manual_boundaries(audio, sr)
    transcription, model_boundaries = get_model_boundaries(audio, sr)
    rmse = rmse_boundaries(manual_boundaries, model_boundaries)

    print("Transcription:", transcription)
    print("Manual boundaries:", manual_boundaries[:10])
    print("Model boundaries:", model_boundaries[:10])
    print("RMSE:", rmse)

    with open(OUTPUT_DIR / "mapping_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Audio file: {input_file}\n")
        f.write(f"Transcription: {transcription}\n")
        f.write(f"Manual boundaries: {manual_boundaries.tolist()}\n")
        f.write(f"Model boundaries: {model_boundaries.tolist()}\n")
        f.write(f"RMSE: {rmse}\n")

    print(f"Saved outputs to: {OUTPUT_DIR}")