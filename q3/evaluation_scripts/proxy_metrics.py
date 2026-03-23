import numpy as np
import soundfile as sf


def load_pair(original_path, transformed_path):
    x, sr1 = sf.read(original_path)
    y, sr2 = sf.read(transformed_path)

    if x.ndim > 1:
        x = np.mean(x, axis=1)
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    return x, y


def snr_proxy(original, transformed):
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - transformed) ** 2) + 1e-10
    return 10 * np.log10(signal_power / noise_power)


if __name__ == "__main__":
    original_path = "outputs/q3/pp_demo/original.wav"
    transformed_path = "outputs/q3/pp_demo/transformed.wav"

    original, transformed = load_pair(original_path, transformed_path)
    snr = snr_proxy(original, transformed)

    print("Proxy SNR:", snr)