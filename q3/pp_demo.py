import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from privacymodule import PrivacyPreservingModule


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "q3_audio"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "q3" / "pp_demo"


def find_first_wav(data_dir):
    wav_files = sorted(data_dir.glob("*.wav"))
    if not wav_files:
        return None
    return wav_files[0]


def load_audio(file_path, max_length=16000):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    audio, sr = sf.read(str(file_path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)

    if len(audio) < max_length:
        audio = np.pad(audio, (0, max_length - len(audio)))
    else:
        audio = audio[:max_length]

    return audio, sr


def run_demo(input_file, output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio, sr = load_audio(input_file)
    x = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

    model = PrivacyPreservingModule(input_dim=len(audio))
    with torch.no_grad():
        transformed = model(x).squeeze(0).numpy()

    sf.write(output_dir / "original.wav", audio, sr)
    sf.write(output_dir / "transformed.wav", transformed, sr)

    print(f"Saved original and transformed audio to: {output_dir}")


if __name__ == "__main__":
    input_file = find_first_wav(DATA_DIR)

    if input_file is None:
        raise FileNotFoundError(
            f"No .wav file found in {DATA_DIR}\n"
            f"Please place at least one WAV file there."
        )

    print(f"Using audio file: {input_file}")
    run_demo(input_file)