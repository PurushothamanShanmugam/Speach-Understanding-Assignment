import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fftpack import dct


def load_audio(file_path):
    audio, sr = sf.read(file_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32), sr


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
    return pad_signal[indices.astype(np.int32, copy=False)]


def real_cepstrum(frame):
    spectrum = np.fft.fft(frame)
    log_mag = np.log(np.abs(spectrum) + 1e-10)
    cepstrum = np.real(np.fft.ifft(log_mag))
    return cepstrum


def detect_voiced_unvoiced(frames):
    labels = []
    low_q_energy_list = []
    high_q_energy_list = []

    for frame in frames:
        cep = real_cepstrum(frame)
        low_q_energy = np.sum(np.abs(cep[:20]))
        high_q_energy = np.sum(np.abs(cep[20:80]))

        low_q_energy_list.append(low_q_energy)
        high_q_energy_list.append(high_q_energy)

    low_q_energy_list = np.array(low_q_energy_list)
    high_q_energy_list = np.array(high_q_energy_list)

    ratio = high_q_energy_list / (low_q_energy_list + 1e-10)
    threshold = np.median(ratio)

    for r in ratio:
        labels.append(1 if r > threshold else 0)  # 1 = voiced, 0 = unvoiced

    return np.array(labels), low_q_energy_list, high_q_energy_list, ratio


def plot_results(signal, sr, labels, frame_stride, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    t = np.arange(len(signal)) / sr
    frame_times = np.arange(len(labels)) * frame_stride

    plt.figure(figsize=(12, 5))
    plt.plot(t, signal, label="Waveform")
    for i, label in enumerate(labels):
        color = "green" if label == 1 else "red"
        plt.axvspan(frame_times[i], frame_times[i] + frame_stride, color=color, alpha=0.15)
    plt.title("Voiced / Unvoiced Segmentation")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "voiced_unvoiced_segmentation.png"))
    plt.close()


if __name__ == "__main__":
    input_file = "data/q1_audio/sample1.wav"
    out_dir = "outputs/q1/voiced_unvoiced"
    frame_size = 0.025
    frame_stride = 0.010

    signal, sr = load_audio(input_file)
    frames = framing(signal, sr, frame_size, frame_stride)
    labels, low_q, high_q, ratio = detect_voiced_unvoiced(frames)

    print("Total frames:", len(labels))
    print("Voiced frames:", np.sum(labels == 1))
    print("Unvoiced frames:", np.sum(labels == 0))

    plot_results(signal, sr, labels, frame_stride, out_dir)
    print(f"Saved outputs to: {out_dir}")