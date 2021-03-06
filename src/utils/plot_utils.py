import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def multiple_formatter(denominator=2, number=np.pi, latex="\pi"):
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int32(np.rint(den * x / number))
        com = gcd(num, den)
        (num, den) = (int(num / com), int(den / com))
        if den == 1:
            if num == 0:
                return r"$0$"
            if num == 1:
                return r"$%s$" % latex
            elif num == -1:
                return r"$-%s$" % latex
            else:
                return r"$%s%s$" % (num, latex)
        else:
            if num == 1:
                return r"$\frac{%s}{%s}$" % (latex, den)
            elif num == -1:
                return r"$\frac{-%s}{%s}$" % (latex, den)
            else:
                return r"$\frac{%s%s}{%s}$" % (num, latex, den)

    return _multiple_formatter


def plot_fft(audio, fs, title="output spectrum"):
    _, ax = plt.subplots(1, 1, figsize=(10, 4))
    N = len(audio)
    w = scipy.signal.windows.hann((N))
    w_fft = scipy.fft.fft(w * audio)
    xf = scipy.fft.fftfreq(N, 1 / fs)[: N // 2]
    ax.plot(xf, 2.0 / N * np.abs(w_fft[0 : N // 2]), label="fft hann window")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_magnitude_response(f, H, label="magnitude", c="tab:blue", title=""):

    ax = plt.subplot(111)
    ax.plot(f, H)
    ax.semilogx(f, H, label=label, color=c)
    plt.ylabel("Amplitude [dB]")
    plt.xlabel("Frequency [hz]")
    plt.title(title + "Magnitude response")


def plot_phase_response(
    f, angles, mult_locater=(np.pi / 2), denom=2, label="phase", c="tab:blue", title=""
):
    ax = plt.subplot(111)
    plt.plot(f, angles)
    ax.semilogx(f, angles, label=label, color=c)
    plt.ylabel("Angle [radians]")
    plt.xlabel("Frequency [hz]")
    plt.title(title + "Phase response")
    ax.yaxis.set_major_locator(plt.MultipleLocator(mult_locater))
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(multiple_formatter(denominator=denom))
    )


def plot_bode(x, fs):
    f, H, angles = freqz(x, fs)
    # Plot the magnitude response of a signal x.
    plot_magnitude_response(f, H)
    plt.show()
    # phase response
    plot_phase_response(f, angles)
    plt.show()


def plot_ltspice_bode(filename, mult_locater=(np.pi / 2), denom=2):

    f, H_db, angles = ltspice_freqz(filename)
    plot_magnitude_response(f, H_db)
    plt.show()

    plot_phase_response(f, angles, mult_locater=mult_locater, denom=denom)
    plt.show()


def compare_freqz_vs_spice(
    x, fs, spicepath, mult_locater=(np.pi / 2), denom=2, title=""
):
    wdf_f, wdf_H, wdf_angles = freqz(x, fs)
    spice_f, spice_H, spice_angles = ltspice_freqz(spicepath)
    plot_magnitude_response(wdf_f, wdf_H, label="wdf", c="tab:blue")
    plot_magnitude_response(
        spice_f, spice_H, label="spice", c="tab:orange", title=title + " "
    )
    plt.legend()
    plt.show()

    plot_phase_response(wdf_f, wdf_angles, label="wdf", c="tab:blue")
    plot_phase_response(
        spice_f,
        spice_angles,
        label="spice",
        mult_locater=mult_locater,
        denom=denom,
        c="tab:orange",
        title=title + " ",
    )
    plt.legend()
    plt.show()
    # return mean_squared_error(spice_H,wdf_H),mean_squared_error(spice_angles,wdf_angles)


def plot_freqz(x: np.ndarray, fs: int, title: str = "Frequency response"):
    # Plot the frequency response of a signal x.
    w, h = scipy.signal.freqz(x, 1, 2**13)
    # w, h = signal.freqz(x, 1)
    magnitude = 20 * np.log10(np.abs(h) + np.finfo(float).eps)
    phase = np.angle(h)
    magnitude_peak = np.max(magnitude)
    top_offset = 10
    bottom_offset = 70
    frequencies = w / (2 * np.pi) * fs

    _, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 5))
    # Make plot pretty.
    xlims = [10**0, 10 ** np.log10(fs / 2)]
    ax[0].semilogx(frequencies, magnitude, label="WDF")
    ax[0].set_xlim(xlims)
    ax[0].set_ylim([magnitude_peak - bottom_offset, magnitude_peak + top_offset])
    ax[0].set_xlabel("Frequency [Hz]")
    ax[0].set_ylabel("Magnitude [dBFs]")

    # plt.set_xlim([np.min(f), np.max(f)])
    phase = 180 * phase / np.pi
    ax[1].semilogx(frequencies, phase, label="WDF")
    ax[1].set_xlim(xlims)
    ax[1].set_ylim([-180, 180])
    ax[1].set_xlabel("Frequency [Hz]")
    ax[1].set_ylabel("Phase [degrees]")

    plt.suptitle(title)
    plt.show()


def compare_waveforms(x, y, out_idx, title="", x_label="input", y_label="output"):
    _, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(x[:out_idx], label=x_label)
    ax.plot(y[:out_idx], label=y_label)
    ax.set_xlim([0, out_idx - 1])
    plt.title(title)
    plt.legend()
    plt.show()


def freqz(x, fs):
    w, h = scipy.signal.freqz(x, 1, 4096)
    H = 20 * np.log10(np.abs(h))
    f = w / (2 * np.pi) * fs
    angles = np.unwrap(np.angle(h))
    return f, H, angles


def ltspice_freqz(filename, out_label="V(vout)"):
    def imag_to_mag(z):
        a, b = map(float, z.split(","))
        return 20 * np.log10(np.sqrt(a * a + b * b))

    def imag_to_phase(z):
        a, b = map(float, z.split(","))
        return np.arctan2(b, a)

    x = pd.read_csv(filename, delim_whitespace=True)
    x["H_dB"] = x[out_label].apply(imag_to_mag)
    x["Phase"] = x[out_label].apply(imag_to_phase)
    f = np.array(x["Freq."])
    H_db = np.array(x["H_dB"])
    angles = np.array(x["Phase"])
    return f, H_db, angles


def get_freqz_error_vs_spice(x,fs,spice_path,error='mse'):

    wdf_f, wdf_H, wdf_angles = freqz(x, fs)
    spice_f, spice_H, spice_angles = ltspice_freqz(spice_path)

    wdf_magnitude = dict(zip(wdf_f, wdf_H))
    wdf_phase = dict(zip(wdf_f, wdf_angles))

    pred_mag = np.zeros(len(spice_f))
    true_mag = np.zeros(len(spice_f))
    pred_phase = np.zeros(len(spice_f))
    true_phase = np.zeros(len(spice_f))
    pred_f = np.zeros(len(spice_f))
    true_f = np.zeros(len(spice_f))
    
    def get_closest(d, search_key):
        if not d.get(search_key):
            key = min(d.keys(), key=lambda key: abs(key - search_key))
            return key, d[key]
        return search_key, d[search_key]

    for i in range(len(spice_f)):
        # find nearest frequencies
        closest, mag_val = get_closest(wdf_magnitude, spice_f[i])
        phase_val = get_closest(wdf_phase, spice_f[i])[1]
        pred_mag[i] = mag_val
        true_mag[i] = spice_H[i]
        pred_phase[i] = phase_val
        true_phase[i] = spice_angles[i]
        pred_f[i] = closest
        true_f[i]= spice_f[i]
    if error == 'mse':
        return (
            mean_squared_error(true_mag,pred_mag),
            mean_squared_error(true_phase,pred_phase),
            mean_squared_error(true_f,pred_f)
        )
    elif error == 'euclidean':
        return (
            np.linalg.norm(true_mag-pred_mag),
            np.linalg.norm(true_phase-pred_phase),
            np.linalg.norm(true_f-pred_f)
        )
    