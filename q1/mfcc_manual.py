import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.fftpack import dct
from scipy.signal import get_window


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "q1_audio"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "q1" / "mfcc"


# -----------------------------
# Find first WAV file
# -----------------------------
def find_first_wav(data_dir):
    wav_files = sorted(data_dir.glob("*.wav"))
    if not wav_files:
        return None
    return wav_files[0]


# -----------------------------
# Load audio
# -----------------------------
def load_audio(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {file_path}\n"
            f"Please place at least one .wav file inside: {DATA_DIR}"
        )

    audio, sr = sf.read(str(file_path))

    # convert stereo → mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return audio.astype(np.float32), sr


# -----------------------------
# Pre-emphasis
# -----------------------------
def pre_emphasis(signal, alpha=0.97):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])


# -----------------------------
# Framing
# -----------------------------
def framing(signal, sr, frame_size=0.025, frame_stride=0.010):
    frame_length = int(round(frame_size * sr))
    frame_step = int(round(frame_stride * sr))
    signal_length = len(signal)

    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length,))
    pad_signal = np.append(signal, z)

    indices = (
        np.tile(np.arange(0, frame_length), (num_frames, 1))
        + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    )

    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames


# -----------------------------
# Windowing
# -----------------------------
def apply_window(frames, window_type="hamming"):
    win = get_window(window_type, frames.shape[1], fftbins=True)
    return frames * win


# -----------------------------
# Power spectrum
# -----------------------------
def power_spectrum(frames, nfft=512):
    mag_frames = np.absolute(np.fft.rfft(frames, nfft))
    pow_frames = (1.0 / nfft) * (mag_frames ** 2)
    return pow_frames


# -----------------------------
# Mel conversions
# -----------------------------
def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700.0)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


# -----------------------------
# Mel filterbank
# -----------------------------
def mel_filterbank(sr, nfft=512, nfilt=26):
    low_mel = 0
    high_mel = hz_to_mel(sr / 2)

    mel_points = np.linspace(low_mel, high_mel, nfilt + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((nfft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))

    for m in range(1, nfilt + 1):
        f_m_minus = bins[m - 1]
        f_m = bins[m]
        f_m_plus = bins[m + 1]

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus + 1e-10)

        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m + 1e-10)

    return fbank


# -----------------------------
# MFCC pipeline
# -----------------------------
def compute_mfcc(signal, sr):
    emphasized = pre_emphasis(signal)

    frames = framing(emphasized, sr)
    windowed = apply_window(frames)

    pow_frames = power_spectrum(windowed)

    fbank = mel_filterbank(sr)
    filter_banks = np.dot(pow_frames, fbank.T)

    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    log_fbanks = np.log(filter_banks)

    mfcc = dct(log_fbanks, type=2, axis=1, norm="ortho")[:, :13]

    return mfcc


# -----------------------------
# Save plots
# -----------------------------
def save_plots(signal, sr, mfcc):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    time_axis = np.arange(len(signal)) / sr

    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, signal)
    plt.title("Waveform")
    plt.savefig(OUTPUT_DIR / "waveform.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.imshow(mfcc.T, aspect="auto", origin="lower")
    plt.title("MFCC")
    plt.colorbar()
    plt.savefig(OUTPUT_DIR / "mfcc.png")
    plt.close()


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    input_file = find_first_wav(DATA_DIR)

    if input_file is None:
        raise FileNotFoundError(
            f"No WAV file found in {DATA_DIR}\n"
            f"Put at least one .wav file there."
        )

    print(f"Using audio file: {input_file}")

    signal, sr = load_audio(input_file)

    mfcc = compute_mfcc(signal, sr)

    print("Audio length:", len(signal))
    print("Sample rate:", sr)
    print("MFCC shape:", mfcc.shape)

    save_plots(signal, sr, mfcc)

    print(f"Saved outputs to: {OUTPUT_DIR}")