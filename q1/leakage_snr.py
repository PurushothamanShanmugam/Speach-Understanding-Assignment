import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import get_window


def load_audio(file_path):
    audio, sr = sf.read(file_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32), sr


def get_segment(signal, sr, start_sec=0.5, duration_sec=0.03):
    start = int(start_sec * sr)
    end = start + int(duration_sec * sr)
    return signal[start:end]


def compute_fft(segment, window_name, nfft=1024):
    window = get_window(window_name, len(segment), fftbins=True)
    windowed = segment * window
    spectrum = np.fft.rfft(windowed, nfft)
    magnitude = np.abs(spectrum)
    return magnitude


def spectral_leakage_metric(magnitude):
    total_energy = np.sum(magnitude ** 2)
    peak_idx = np.argmax(magnitude)
    left = max(0, peak_idx - 3)
    right = min(len(magnitude), peak_idx + 4)
    main_lobe_energy = np.sum(magnitude[left:right] ** 2)
    leakage_energy = total_energy - main_lobe_energy
    return leakage_energy / (total_energy + 1e-10)


def snr_estimate(original, processed):
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - processed) ** 2) + 1e-10
    return 10 * np.log10(signal_power / noise_power)


def analyze_windows(segment):
    mapping = {
        "rectangular": "boxcar",
        "hamming": "hamming",
        "hanning": "hann",
    }

    results = []
    processed_versions = {}

    for label, window_name in mapping.items():
        window = get_window(window_name, len(segment), fftbins=True)
        processed = segment * window
        processed_versions[label] = processed

        magnitude = compute_fft(segment, window_name)
        leakage = spectral_leakage_metric(magnitude)
        snr = snr_estimate(segment, processed)

        results.append({
            "window": label,
            "leakage": leakage,
            "snr_db": snr
        })

    return results, processed_versions


def plot_spectra(segment, processed_versions, sr, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    nfft = 1024
    freqs = np.fft.rfftfreq(nfft, d=1/sr)

    for label, processed in processed_versions.items():
        spec = np.abs(np.fft.rfft(processed, nfft))
        plt.figure(figsize=(10, 4))
        plt.plot(freqs, spec)
        plt.title(f"Spectrum - {label}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{label}_spectrum.png"))
        plt.close()


if __name__ == "__main__":
    input_file = "data/q1_audio/sample1.wav"
    out_dir = "outputs/q1/leakage_snr"

    signal, sr = load_audio(input_file)
    segment = get_segment(signal, sr)

    results, processed_versions = analyze_windows(segment)
    plot_spectra(segment, processed_versions, sr, out_dir)

    print("Window comparison:")
    for row in results:
        print(row)

    print(f"Saved plots to: {out_dir}")