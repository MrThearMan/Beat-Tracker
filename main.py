"""

Beat tracker for detecting note onsets and creating a pseudo click-track.

Original papers and examples written in MATLAB by Peter Grosche and Meinard Müller: http://resources.mpi-inf.mpg.de/MIR/tempogramtoolbox/

Video on the toolbox: https://www.youtube.com/watch?v=FmwpkdcAXl0

Translated from the original code to Python.

"""

import os
import math
import wave
from typing import Literal

import nnresample
import scipy.signal
import scipy.ndimage
import scipy.io.wavfile

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def round2(n, decimals=0):
    """Round half up for positive and half down for negative numbers.

    :param n: Number to round.
    :type n: int or float
    :param decimals: decimals to round to.
    :type decimals: int
    :return: Rounded number
    :rtype: int or float
    """

    if decimals >= 0:
        n *= (10 ** decimals)
    else:
        raise ValueError("Decimals must be positive or zero.")

    if n > 0:
        val = int(math.floor(n + 0.5))
    else:
        val = int(math.ceil(n - 0.5))

    val /= (10 ** decimals)

    if decimals == 0:
        val = int(val)

    return val


def filter2(H, X, mode="same"):
    """2-D digital finite impulse response filter.

    :param H: Array to apply finite impulse response filter to.
    :type H: np.ndarray or int or float
    :param X: Array of coefficients that define the finite impulse response filter.
    :type X: np.ndarray or int or float
    :param mode: A string indicating the size of the output:
    - 'full' The output is the full discrete linear convolution of the inputs.
    - 'valid' The output consists only of those elements that do not rely on the zero-padding. Either 'X' or 'H' must be at least as large as the other in every dimension.
    - 'same' The output is the same size as 'H', centered with respect to the 'full' output. (Default)
    :type mode: str
    :return: Filtered array.
    :rtype: np.ndarray
    """

    return scipy.signal.convolve2d(X, np.rot90(H, 2), mode=mode)


def read_waveform(file, resample=True, resampling_rate=22050, to_mono=True, mono_convertion_mode: Literal["left_only", "right_only", "downmix"] = "downmix"):
    """Convert audiofile to wanted format.

    :param file: Filepath of filename.
    :type file: str
    :param resample: Resample to resampling rate.
    :type resample: bool
    :param resampling_rate: Wanted sample rate.
    :type resampling_rate: int
    :param to_mono: Convert signal to mono.
    :type to_mono: bool
    :param mono_convertion_mode: 1. read left channel only, 2. read right channel only, or 3. average left and right channels
    :return: Audio data and related info.
    :rtype: tuple[np.ndarray, dict]
    """

    info = dict.fromkeys(["filename", "sampling_rate", "channels", "size", "duration", "bit_depth", "bit_rate"])

    with wave.open(file) as audio:
        info["bit_depth"] = audio.getsampwidth() * 8

    with sf.SoundFile(file) as audio:
        data = audio.read()
        fileformat = audio.format

        info["filename"] = audio.name
        info["sampling_rate"] = audio.samplerate
        info["channels"] = audio.channels
        info["size"] = audio.frames
        info["duration"] = info["size"] / info["sampling_rate"]
        info["bit_rate"] = info["sampling_rate"] * info["bit_depth"] * info["channels"]

    assert fileformat == "WAV", "File must be a WAV-file."

    if info["channels"] == 2 and to_mono:
        info["channels"] = 1

        if mono_convertion_mode == "left_only":
            data = data[:, 0]
        elif mono_convertion_mode == "right_only":
            data = data[:, 1]
        elif mono_convertion_mode == "downmix":
            data = np.mean(data, axis=1)
        else:
            raise ValueError(f"Conversion mode \"{mono_convertion_mode}\" not specified.")

    if resample and info["sampling_rate"] != resampling_rate:
        data: np.ndarray = nnresample.resample(s=data, up=resampling_rate, down=info["sampling_rate"])  # noqa
        info["sampling_rate"] = resampling_rate
        info["size"] = len(data)
        info["duration"] = info["size"] / info["sampling_rate"]
        info["bit_rate"] = info["sampling_rate"] * info["bit_depth"] * info["channels"]

    return data, info


def audio_to_spectrogram_via_STFT(audio, sampling_rate, stft_window=None, stepsize=None, magnitude_only=True):
    """Computes the spectrogram of the audio signal using a STFT.

    :param audio: Audio data.
    :type audio: np.ndarray
    :param sampling_rate: Audio sampling rate.
    :type sampling_rate: int
    :param stft_window: Short-time Fourier-Transform hanning window (np.hanning(4096) by default)
    :type stft_window: np.ndarray
    :param stepsize: Novelty curve window stepsize in frames.
    :type stepsize: int or float
    :param magnitude_only: Return only the complex magnitude of the spectrogram.
    :type magnitude_only: bool
    :return: Complex spectrogram, feature rate of the spectrogram (Hz), vector of frequecncies (Hz) for the coefficients and vector of time positions (sec) for the frames
    :rtype: tuple
    """

    if stft_window is None:
        stft_window = np.hanning(4096)

    window_length = len(stft_window)

    if stepsize is None:
        stepsize = round2(window_length / 2)

    nfft = window_length

    feature_rate = sampling_rate / stepsize
    wav_size = len(audio)
    first_win = math.floor(window_length / 2)
    num_frames = math.ceil(wav_size / stepsize)
    num_coeffs = int(max(nfft, window_length) / 2) + 1
    zeros_to_pad = max(0, nfft - window_length)

    spectogram = np.zeros((num_coeffs, num_frames), dtype=complex if not magnitude_only else np.float64)

    frame = np.subtract(np.array(list(range(1, window_length + 1))), first_win)

    for n in range(0, num_frames):

        num_zeros = sum(frame < 1)
        num_vals = sum(frame > 0)

        if num_zeros > 0:
            x = np.append(np.zeros(num_zeros), audio[0:num_vals]) * stft_window

        elif frame[-1] > wav_size:
            x = np.append(audio[int(frame[0]):wav_size], np.zeros(window_length - (wav_size - int(frame[0]) + 1) + 1)) * stft_window

        else:
            x = audio[int(frame[0]):int(frame[-1]) + 1] * stft_window

        if zeros_to_pad > 0:
            x = np.append(x, np.zeros(zeros_to_pad))

        xs = np.fft.fft(x)

        if magnitude_only:
            spectogram[:, n] = [abs(xs[x]) for x in range(num_coeffs)]
        else:
            spectogram[:, n] = [xs[x] for x in range(num_coeffs)]

        frame = frame + stepsize

    t = np.arange(0, num_frames) * stepsize/sampling_rate
    f = np.arange(0, math.floor(max(nfft, window_length) / 2) + 1) / math.floor(max(nfft, window_length) / 2) * (sampling_rate / 2)
    f = f[:num_coeffs]

    return spectogram, feature_rate, f, t


def compute_fourier_coefficients(signal, window, overlap, frequencies, sampling_rate):
    """Compuses the complex fourier coefficients for the given frequencies in the given signal with the given sampling rate (windowed using "window" and "overlap").

    :param signal: Time domain signal.
    :type signal: np.ndarray
    :param window: Vector containing window function.
    :type window: np.ndarray
    :param overlap: Overlap given in samples.
    :type overlap: int
    :param frequencies: Vector of frequencies to compute of fourier coefficients for (Hz)
    :type frequencies: np.ndarray
    :param sampling_rate: Sampling rate of  the given signal (Hz)
    :type sampling_rate: int or float
    :return: x: complex fourier coefficients, t: time in sec of window positions
    :rtype: tuple[np.ndarray, np.ndarray]
    """

    signal = signal.flatten()
    frequencies = (frequencies / 60).flatten()

    win_len = window.size
    hopsize = win_len - overlap

    two_pi_t = 2 * np.pi * (np.arange(0, win_len) / sampling_rate)
    win_num = int(np.fix((signal.size - overlap) / hopsize))

    coefficients = np.zeros((win_num, frequencies.size), dtype=complex)
    t = np.arange(win_len / 2, signal.size - win_len / 2 + 1, hopsize) / sampling_rate

    for f0 in range(0, frequencies.size):

        two_pi_ft = frequencies[f0] * two_pi_t
        cosine = np.cos(two_pi_ft)
        sine = np.sin(two_pi_ft)

        for w in range(0, win_num):
            start = w * hopsize
            stop = start + win_len

            sig = signal[start:stop] * window
            co = float(np.sum(sig * cosine))
            si = float(np.sum(sig * sine))

            coefficients[w, f0] = complex(real=co, imag=si)

    coefficients = coefficients.T

    return coefficients, t


def audio_to_novelty_curve(audio, sampling_rate, threshold=-74, window_length=None, stepsize=None, log_compression=True, compression_constant=1000, resample_feature_rate=200):
    """Compute spectrogram

    :param audio: Audio data.
    :type audio: np.ndarray
    :param sampling_rate: Sampling rate of the audio (Hz)
    :type sampling_rate: float or int
    :param threshold: Threshold for the normalization (dB)
    :type threshold: int
    :param window_length: Lenght of the stft window.
    :type window_length: float
    :param stepsize: Stepsize for the STFT.
    :type stepsize: float
    :param log_compression: Enable/disable log compression.
    :type log_compression: bool
    :param compression_constant: Constant for log compression
    :type compression_constant: int
    :param resample_feature_rate: Feature rate of the resulting novelty curve (resampled, independent of stepsize)
    :type resample_feature_rate: int
    :return:
    :rtype:
    """

    if window_length is None:
        window_length: float = 1024 * sampling_rate / 22050

    if stepsize is None:
        stepsize: float = 512 * sampling_rate / 22050

    stft_window = np.hanning(round2(window_length))

    # Compute spectrogram
    spec_data, feature_rate, _, _ = audio_to_spectrogram_via_STFT(audio, sampling_rate, stft_window=stft_window, stepsize=stepsize)

    # Normalize and convert to dB
    threshold = 10 ** (threshold / 20)
    spec_data = spec_data / np.amax(spec_data)
    with np.nditer(spec_data, op_flags=["readwrite"]) as it:
        for x in it:
            x[...] = max(x, threshold)

    # bandwise processing
    bands = np.array([[0, 500], [500, 1250], [1250, 3125], [3125, 7812.5], [7812.5, math.floor(sampling_rate / 2)]])

    band_novelty_curves = np.zeros((len(bands), spec_data.shape[1]))

    for band in range(0, len(bands)):

        bins = np.around(bands[band] / (sampling_rate / window_length))

        bins[0] = max(0, bins[0])
        bins[1] = min(len(spec_data) - 1, bins[1])

        # Band novelty curve
        band_data = spec_data[int(bins[0]):int(bins[1]), :]

        if log_compression and compression_constant > 0:
            band_data = np.log(1 + band_data * compression_constant) / (np.log(1 + compression_constant))

        # Smoothed differentiator
        diff_len = 0.3
        diff_len = max(math.ceil(diff_len * sampling_rate / stepsize), 5)
        diff_len = 2 * round2(diff_len / 2) + 1

        diff_filter = np.hanning(diff_len) * np.concatenate((-1 * np.ones(math.floor(diff_len / 2)), np.array([0]), np.ones(math.floor(diff_len / 2))))
        diff_filter = diff_filter[..., None].T

        rm1 = np.array([band_data[:, 0] for _ in range(math.floor(diff_len / 2))]).T
        rm2 = np.array([band_data[:, -1] for _ in range(math.floor(diff_len / 2))]).T
        hhh = np.concatenate((rm1, band_data, rm2), axis=1)

        band_diff = filter2(diff_filter, hhh)
        band_diff[band_diff < 0] = 0
        band_diff = band_diff[:, math.floor(diff_len / 2) - 1: -1 - math.floor(diff_len / 2)]

        # Normalize band
        norm_len = 5
        norm_len = max(math.ceil(norm_len * sampling_rate / stepsize), 3)
        norm_filter = np.hanning(norm_len)[..., None]
        norm_curve = filter2(norm_filter / np.sum(norm_filter), np.sum(band_data, axis=0)[..., None]).T

        # Boundary correction
        norm_filter_sum = ((np.sum(norm_filter) - np.cumsum(norm_filter, axis=0)) / np.sum(norm_filter)).T

        norm_curve[:, 0:math.floor(norm_len / 2)] = norm_curve[:, 0:math.floor(norm_len / 2)] / np.fliplr(norm_filter_sum[:, 0:math.floor(norm_len / 2)])
        norm_curve[:, -math.floor(norm_len / 2):] = norm_curve[:, -math.floor(norm_len / 2):] / norm_filter_sum[:, 0:math.floor(norm_len / 2)]

        band_diff /= norm_curve

        # Compute novelty curve of band
        band_novelty_curves[band, :] = np.sum(band_diff, axis=0)

    novelty_curve = np.mean(band_novelty_curves, axis=0)

    # Resample curve
    if resample_feature_rate > 0 and resample_feature_rate != feature_rate:
        novelty_curve: np.ndarray = nnresample.resample(novelty_curve, resample_feature_rate, int(feature_rate))  # noqa
        novelty_curve = novelty_curve[..., None]
        feature_rate = resample_feature_rate

    # Average subtraction
    smooth_len: int = max(int(math.ceil(1.5 * sampling_rate / stepsize)), 3)
    smooth_filter: np.ndarray = np.hanning(smooth_len)[..., None]

    local_average: np.ndarray = filter2(smooth_filter / np.sum(smooth_filter), novelty_curve)

    novelty_curve = novelty_curve - local_average
    novelty_curve[novelty_curve < 0] = 0

    return novelty_curve, feature_rate


def novelty_curve_to_tempogram_via_DFT(novelty_curve, feature_rate, tempo_window=6, stepsize=None, bpm=None):
    """Computes a complex valued fourier tempogram for a given novelty curve indicating note onset candidates in the form of peaks.

    :param novelty_curve: Novelty curve-array.
    :type novelty_curve: np.ndarray
    :param feature_rate: Feature rate of the novelty curve (Hz).
    :type feature_rate: int or float
    :param stepsize: Novelty curve window stepsize in frames.
    :type stepsize: int
    :param tempo_window: Trade-off between time- and tempo resolution.
    Lower tempo window means tempo will be more susceptible to changes but more accurate and higher means less tempo changes but more stable tempo. Default is a compromise of both.
    :type tempo_window:
    :param bpm: Vector containing BPM values to compute. Prior knowledge can help to stablilize the tempo and select the correct predominant pulse level (Measure, Tactus, Tatum..)
    :type bpm: np.ndarray
    :return: tempogram: Complex valued fourier tempogram, bpm: Vector of BPM values of the tempo axis of the tempogram, tt: Vector of time positions (in sec) for the frames of the tempogram
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """

    if bpm is None:
        bpm = np.arange(30, 601, 1)

    if stepsize is None:
        stepsize = math.ceil(feature_rate / 5)

    win_len = round2(tempo_window * feature_rate)
    win_len = win_len + (win_len % 2) - 1

    window_tempogram = np.hanning(win_len)

    novelty_curve = np.concatenate((np.zeros((round2(win_len / 2), 1)), novelty_curve, np.zeros((round2(win_len / 2), 1))), axis=0)

    tempogram, t = compute_fourier_coefficients(signal=novelty_curve, frequencies=bpm, window=window_tempogram, overlap=win_len - stepsize, sampling_rate=feature_rate)

    tempogram = tempogram / math.sqrt(win_len) / np.sum(window_tempogram) * win_len
    bpm = bpm[..., None]
    t = (t - t[0])[..., None].T

    return tempogram, bpm, t


def normalize_curve_to_threshold(curve, threshold=0.0001):
    """Normalizes a curve and replaces vectors below the given threshold with unit vectors.

    :param curve: Curve to normalize.
    :type curve: np.ndarray
    :param threshold: Threshold for an unit vector.
    :type threshold: float
    :return: Nomalized curve.
    :rtype: np.ndarray
    """

    normalized = np.zeros(curve.shape, dtype=complex)

    unit_vec = np.ones((curve.shape[0], 1), dtype=complex)
    unit_vec = unit_vec / np.linalg.norm(unit_vec, 2)

    for k in range(0, curve.shape[1]):
        n = np.linalg.norm(curve[:, k], 2)

        if n < threshold:
            normalized[:, k] = unit_vec
        else:
            normalized[:, k] = curve[:, k] / n

    return normalized


def tempogram_to_PLP_curve(tempogram, feature_rate, tt, bpm, stepsize=0, tempo_window=6, tempocurve=None):
    """Computes a PLP curve from a complex tempogram representatio. Uses maximum values for each frame (predominant local periodicity) or a given tempocurve.

    :param tempogram: Tempogram-array.
    :type tempogram: np.ndarray
    :param feature_rate: Feature rate of the novelty curve used to make the tempogram.
    :type feature_rate: int or float
    :param tt: Vector of time positions (in sec) for the frames of the tempogram.
    :type tt: np.ndarray
    :param bpm: Vector containing BPM values to compute.
    :type bpm: np.ndarray
    :param stepsize: Window stepsize in frames (of novelty curve)
    :type stepsize: int
    :param tempo_window: Trade-off between time- and tempo resolution.
    Lower tempo window means tempo will be more susceptible to changes but more accurate and higher means less tempo changes but more stable tempo. Default is a compromise of both.
    :type tempo_window: int
    :param tempocurve: Optional tempocurve (in BPM), one entry for each tempogram frame.
    :type tempocurve: np.ndarray or None
    :return: Calculated plp-curve.
    :rtype: np.ndarray
    """

    if stepsize == 0:
        stepsize = math.ceil(feature_rate / 5)

    tempogram_abs = np.abs(tempogram)
    local_max = np.zeros(tempogram_abs.shape[1], dtype=int)

    if tempocurve:
        for frame in range(0, tempogram_abs.shape[1]):
            local_max[frame] = int(np.argmin(np.abs(bpm - tempocurve[frame])))
    else:
        for frame in range(0, tempogram_abs.shape[1]):
            local_max[frame] = int(np.argmax(tempogram_abs[:, frame]))

    win_len = round2(tempo_window * feature_rate)
    win_len = win_len + (win_len % 2) - 1

    t = (tt * feature_rate)[0]

    plp = np.zeros((tempogram.shape[1]) * stepsize)

    window = np.hanning(win_len)
    window = window / (np.sum(window) / win_len)
    window = window / (win_len / stepsize)

    for frame in range(0, tempogram.shape[1]):

        t0 = math.ceil(t[frame] - win_len / 2)
        t1 = math.floor(t[frame] + win_len / 2)

        phase = -np.angle(tempogram[local_max[frame], frame])
        t_period = feature_rate * 60 / bpm[local_max[frame]]
        p_len = (t1 - t0 + 1) / t_period

        cosine = window * np.cos(np.arange(0, p_len, 1 / t_period) * 2 * math.pi + phase)[:window.size]

        if t0 < 1:
            cosine = cosine[-t0 + 1:]
            t0 = 0
        else:
            t0 -= 1

        if t1 > plp.shape[0]:
            cosine = cosine[:plp.shape[0] - t1]
            t1 = plp.shape[0]

        plp[t0:t1] = plp[t0:t1] + cosine

    plp[plp < 0] = 0

    return plp


def sonify(curve, audio, sampling_rate, feature_rate, min_confidence=0.01, only_ticks=False, confidence=True, half_tempo=False, half_tempo_start=1):
    """Mix the peaks of the given curve as clicks with the given audio.

    :param curve: Curve to sonify (should be normalised to 0-1 range)
    :type curve: np.ndarray
    :param audio: audio to mix the sonified curve with
    :type audio: np.ndarray
    :param sampling_rate: sampling rate of the curve and audio
    :type sampling_rate: int or float
    :param feature_rate: feature rate of the curve and audio
    :type feature_rate: int or float
    :param min_confidence: Minimum confidence require for the peaks.
    :type min_confidence: float
    :param only_ticks: sonification will only contain the ticks and not the audio.
    :type only_ticks: bool
    :param confidence: Tick volume is determided by the value of the curve (curve should be normalised to 0-1 range)
    :type confidence: bool
    :param half_tempo: Remove every other peak for half tempo.
    :type half_tempo: bool
    :param half_tempo_start: Start removing peaks for half tempo from this peak onwards. All peaks before this peak are removed.
    :type half_tempo_start: int
    :return: Sonified curve.
    :rtype: np.ndarray
    """

    pos = np.append(curve, curve[-1]) > np.insert(curve, 0, curve[0])
    neg = ~pos

    peaks = np.where(pos[:pos.shape[0] - 1] * neg[1:])[0]

    if half_tempo:
        peaks = peaks[half_tempo_start::2]

    values = curve[peaks].flatten()
    values = values / np.max(values)

    # Remove small peaks
    peaks = peaks[values >= min_confidence]
    values = values[values >= min_confidence]

    click = np.array([
        0.0000,
        0.0000,
        -0.0001,
        0.0002,
        -0.0001,
        0.0001,
        -0.0000,
        -0.0001,
        0.0001,
        0.0938,
        0.1861,
        0.2755,
        0.3606,
        0.4400,
        0.5125,
        0.5769,
        0.6324,
        0.6778,
        0.7127,
        0.7362,
        0.7484,
        0.7487,
        0.7373,
        0.7143,
        0.6800,
        0.6352,
        0.5803,
        0.5164,
        0.4442,
        0.3654,
        0.2803,
        0.1915,
        0.0989,
        0.0054,
        -0.0885,
        -0.1810,
        -0.2704,
        -0.3560,
        -0.4356,
        -0.5086,
        -0.5734,
        -0.6295,
        -0.6755,
        -0.7108,
        -0.7354,
        -0.7479,
        -0.7491,
        -0.7382,
        -0.7159,
        -0.6823,
        -0.6380,
        -0.5837,
        -0.5202,
        -0.4485,
        -0.3700,
        -0.2853,
        -0.1965,
        -0.1043,
        -0.0107,
        0.0832,
        0.1758,
        0.2655,
        0.3511,
        0.4314,
        0.5045,
        0.5702,
        0.6264,
        0.6733,
        0.7091,
        0.7343,
        0.7475,
        0.7493,
        0.7392,
        0.7174,
        0.6845,
        0.6408,
        0.5871,
        0.5240,
        0.4529,
        0.3744,
        0.2905,
        0.2014,
        0.1098,
        0.0159,
        -0.0779,
        -0.1704,
        -0.2606,
        -0.3464,
        -0.4269,
        -0.5007,
        -0.5665,
        -0.6235,
        -0.6709,
        -0.7073,
        -0.7332,
        -0.7469,
        -0.7497,
        -0.7399,
        -0.7191,
        -0.6867,
        -0.6434,
        -0.5905,
        -0.5278,
        -0.4571,
        -0.3792,
        -0.2952,
        -0.2068,
        -0.1148,
        -0.0215,
        0.0726,
        0.1653,
        0.2556,
        0.3416,
        0.4225,
        0.4966,
        0.5631,
        0.6206,
        0.6683,
        0.7058,
        0.7318,
        0.7467,
        0.7497,
        0.7408,
        0.7206,
        0.6888,
        0.6463,
        0.5937,
        0.5316,
        0.4613,
        0.3838,
        0.3002,
        0.2119,
        0.1202,
        0.0268,
        -0.0673,
        -0.1600,
        -0.2506,
        -0.3368,
        -0.4182,
        -0.4926,
        -0.5595,
        -0.6176,
        -0.6658,
        -0.7039,
        -0.7307,
        -0.7461,
        -0.7499,
        -0.7416,
        -0.7220,
        -0.6909,
        -0.6488,
        -0.5971,
        -0.5352,
        -0.4656,
        -0.3883,
        -0.3051,
        -0.2171,
        -0.1254,
        -0.0322,
        0.0620,
        0.1548,
        0.2454,
        0.3322,
        0.4135,
        0.4887,
        0.5558,
        0.6147,
        0.6633,
        0.7021,
        0.7294,
        0.7456,
        0.7499,
        0.7425,
        0.7234,
        0.6929,
        0.6517,
        0.6000,
        0.5392,
        0.4696,
        0.3930,
        0.3099,
        0.2220,
        0.1308,
        0.0374,
        -0.0566,
        -0.1497,
        -0.2403,
        -0.3275,
        -0.4090,
        -0.4846,
        -0.5523,
        -0.6114,
        -0.6610,
        -0.7001,
        -0.7283,
        -0.7449,
        -0.7499,
        -0.7432,
        -0.7248,
        -0.6949,
        -0.6544,
        -0.6032,
        -0.5429,
        -0.4739,
        -0.3974,
        -0.3149,
        -0.2271,
        -0.1362,
        -0.0426,
        0.0513,
        0.1443,
        0.2355,
        0.3224,
        0.4048,
        0.4804,
        0.5486,
        0.6084,
        0.6583,
        0.6982,
        0.7269,
        0.7444,
        0.7500,
        0.7439,
        0.7261,
        0.6970,
        0.6569,
        0.6064,
        0.5466,
        0.4778,
        0.4022,
        0.3194,
        0.2324,
        0.1413,
        0.0479,
        -0.0457,
        -0.1393,
        -0.2302,
        -0.3177,
        -0.4002,
        -0.4762,
        -0.5451,
        -0.6051,
        -0.6559,
        -0.6962,
        -0.7256,
        -0.7437,
        -0.7500,
        -0.7445,
        -0.7276,
        -0.6988,
        -0.6595,
        -0.6095,
        -0.5501,
        -0.4822,
        -0.4064,
        -0.3244,
        -0.2374,
        -0.1464,
        -0.0536,
        0.0407,
        0.1338,
        0.2252,
        0.3128,
        0.3957,
        0.4722,
        0.5413,
        0.6021,
        0.6532,
        0.6942,
        0.7242,
        0.7430,
        0.7499,
        0.7452,
        0.7287,
        0.7009,
        0.6619,
        0.6128,
        0.5537,
        0.4862,
        0.4109,
        0.3292,
        0.2424,
        0.1517,
        0.0587,
        -0.0353,
        -0.1286,
        -0.2201,
        -0.3079,
        -0.3911,
        -0.4680,
        -0.5377,
        -0.5988,
        -0.6506,
        -0.6921,
        -0.7229,
        -0.7421,
        -0.7499,
        -0.7457,
        -0.7300,
        -0.7027,
        -0.6645,
        -0.6157,
        -0.5574,
        -0.4902,
        -0.4155,
        -0.3340,
        -0.2475,
        -0.1570,
        -0.0640,
        0.0298,
        0.1235,
        0.2148,
        0.3031,
        0.3865,
        0.4639,
        0.5338,
        0.5957,
        0.6477,
        0.6902,
        0.7213,
        0.7414,
        0.7498,
        0.7462,
        0.7313,
        0.7045,
        0.6670,
        0.6187,
        0.5609,
        0.4943,
        0.4198,
        0.3389,
        0.2525,
        0.1622,
        0.0693,
        -0.0245,
        -0.1181,
        -0.2098,
        -0.2982,
        -0.3820,
        -0.4597,
        -0.5301,
        -0.5923,
        -0.6452,
        -0.6879,
        -0.7199,
        -0.7405,
        -0.7496,
        -0.7469,
        -0.7323,
        -0.7065,
        -0.6693,
        -0.6218,
        -0.5644,
        -0.4984,
        -0.4241,
        -0.3438,
        -0.2574,
        -0.1675,
        -0.0747,
        0.0194,
        0.1126,
        0.2049,
        0.2932,
        0.3773,
        0.4554,
        0.5263,
        0.5891,
        0.6424,
        0.6859,
        0.7184,
        0.7397,
        0.7495,
        0.7472,
        0.7336,
        0.7081,
        0.6717,
        0.6249,
        0.5677,
        0.5025,
        0.4284,
        0.3485,
        0.2624,
        0.1727,
        0.0800,
        -0.0139,
        -0.1076,
        -0.1995,
        -0.2884,
        -0.3727,
        -0.4511,
        -0.5226,
        -0.5857,
        -0.6397,
        -0.6836,
        -0.7168,
        -0.7389,
        -0.7490,
        -0.7478,
        -0.7346,
        -0.7099,
        -0.6742,
        -0.6276,
        -0.5716,
        -0.5061,
        -0.4331,
        -0.3530,
        -0.2676,
        -0.1778,
        -0.0853,
        0.0086,
        0.1022,
        0.1945,
        0.2834,
        0.3681,
        0.4469,
        0.5187,
        0.5824,
        0.6369,
        0.6814,
        0.7152,
        0.7379,
        0.7487,
        0.7483,
        0.7355,
        0.7117,
        0.6764,
        0.6306,
        0.5749,
        0.5101,
        0.4155,
        0.3219,
        0.2317,
        0.1463,
        0.0680,
        -0.0022,
        -0.0631,
        -0.1134,
        -0.1532,
        -0.1817,
        -0.1991,
        -0.2060,
        -0.2026,
        -0.1902,
        -0.1699,
        -0.1427,
        -0.1106,
        -0.0748,
        -0.0375,
        0.0000,
        -0.0001,
        0.0000,
        0.0001,
        -0.0001,
        0.0002,
        -0.0002,
        0.0001,
        0
    ])
    click = click * np.arange(start=1, stop=1 / len(click), step=-1 / len(click)) ** 2
    click: np.ndarray = nnresample.resample(click, up=sampling_rate, down=88200)  # noqa

    out = np.zeros_like(audio)

    for idx in range(0, len(peaks)):
        start = int(np.floor(peaks[idx] / feature_rate * sampling_rate))
        stop = start + len(click)

        if stop <= len(out):
            if confidence:
                out[start:stop] = out[start:stop] + click * values[idx]
            else:
                out[start:stop] = out[start:stop] + click

    if only_ticks:
        return np.concatenate((out[..., None], out[..., None]), axis=1)
    else:
        return np.concatenate((audio[..., None], out[..., None]), axis=1)


def plot_novelty_curve(novelty_curve, duration, ticks=10):
    """Plot novelty curve.

    :param novelty_curve: Novelty curve data.
    :type novelty_curve: np.ndarray
    :param duration: Novelty curve audio duration (s)
    :type duration: int or float
    :param ticks: Number of ticks to show for secons.
    :type ticks: int
    """

    fig, ax = plt.subplots()

    x_ticks = np.arange(start=0, stop=len(novelty_curve) + len(novelty_curve) / ticks, step=len(novelty_curve) / ticks)
    x_labels = np.arange(start=0, stop=round2(duration) + round2(duration) / ticks, step=round2(duration) / ticks)
    x_labels = np.round(x_labels, 1)

    ax.plot(novelty_curve)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_title("Novelty Curve")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Novelty")
    plt.show()


def plot_tempogram(tempogram, bpm, tt):
    """Plot tempogram.

    :param tempogram: Tempogram data.
    :type tempogram: np.ndarray
    """

    fig, ax = plt.subplots()

    ax.imshow(np.abs(tempogram), aspect="auto", extent=[tt[0][0], tt[0][-1], bpm[0][0], bpm[-1][0]], cmap="hot", origin="lower")
    ax.set_title("Tempogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tempo (BPM)")
    plt.show()


def plot_plp_curve(plp, duration, ticks=10):
    """Plot PLP curve.

    :param plp: PLP data.
    :type plp: np.ndarray
    :param duration: Novelty curve audio duration (s)
    :type duration: int or float
    :param ticks: Number of ticks to show for secons.
    :type ticks: int
    """

    fig, ax = plt.subplots()

    x_ticks = np.arange(start=0, stop=len(plp) + len(plp) / ticks, step=len(plp) / ticks)
    x_labels = np.arange(start=0, stop=round2(duration) + round2(duration) / ticks, step=round2(duration) / ticks)
    x_labels = np.round(x_labels, 1)

    ax.plot(plp)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_title("PLP Curve")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Confidence")
    plt.show()




if __name__ == "__main__":

    file1 = "Schumann - Op. 15, No. 03.wav"
    file2 = "Poulenc - Valse.wav"
    file3 = "Faure - Op. 15, No. 01.wav"
    file4 = "Debussy - Sonata Violin Piano G Minor - 2nd Movement.wav"

    audio, audioinfo = read_waveform(os.path.join("audio", file1))

    novelty_curve, feature_rate = audio_to_novelty_curve(audio, sampling_rate=audioinfo["sampling_rate"])

    # plot_novelty_curve(novelty_curve, audioinfo["duration"])

    tempo_window = 12

    tempogram, bpm, tt = novelty_curve_to_tempogram_via_DFT(novelty_curve=novelty_curve, feature_rate=feature_rate, tempo_window=tempo_window)
    tempogram = normalize_curve_to_threshold(curve=tempogram)

    # plot_tempogram(tempogram, bpm, tt)

    plp = tempogram_to_PLP_curve(tempogram=tempogram, feature_rate=feature_rate, tt=tt, bpm=bpm, tempo_window=tempo_window)
    plp = plp[:novelty_curve.shape[0]][..., None]

    # plot_plp_curve(plp, audioinfo["duration"])

    sonification = sonify(curve=plp, audio=audio, sampling_rate=audioinfo["sampling_rate"], feature_rate=feature_rate, half_tempo=True)

    scipy.io.wavfile.write("sonification.wav", audioinfo["sampling_rate"], sonification)



