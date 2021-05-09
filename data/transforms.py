"""
Module: transforms.py
Authors: Christian Bergler, Hendrik Schroeter
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 12.12.2019
"""

import io
import os
import sys
import math
import resampy  # python module for efficient time-series resampling
import numpy as np
import scipy.fftpack
import soundfile as sf # load audio files

import torch
import torch.nn.functional as F

from typing import List
from multiprocessing import Lock
from utils.FileIO import AsyncFileReader, AsyncFileWriter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

""" 
Load audio file
called by class Dataset(AudioDataset) in audiodataset.py  
"""
def load_audio_file(file_name, sr=None, mono=True):
    # returns data and sr
    # always_2d = True --> 2D array is returned even if mono
    y, sr_orig = sf.read(file_name, always_2d=True, dtype="float32") # y=data, 2D np array (frames x chns), channels: 1st dim, as cols
    if mono and y.ndim == 2 and y.shape[1] > 1: # channels = data.shape[1]; for consistent indexing, mono (frames, 1);
        y = np.mean(y, axis=1, keepdims=True) # stereo -> mono, still a 2D array (frames x 1)
    if sr is not None and sr != sr_orig:
        y = resampy.resample(y, sr_orig, sr, axis=0, filter="kaiser_best") # if wrong SR: resample with Kaiser-window
    return torch.from_numpy(y).float().t() # transpose, (1 x frames) --> needed shape for STFT

"""Composes several transforms to one."""
class Compose(object):

    def __init__(self, *transforms):
        if len(transforms) == 1 and isinstance(transforms[0], list):
            self.transforms = transforms[0]
        else:
            self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

"""Squeezes the given tensor at dim=0."""
class SqueezeDim0(object):

    def __call__(self, x):
        return x.squeeze(dim=0)


"""Unsqueezes the given tensor at dim=0."""
class UnsqueezeDim0(object):

    def __call__(self, x):
        return x.unsqueeze(dim=0)


"""Converts a given numpy array to torch.FloatTensor."""
class ToFloatTensor(object):

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        elif isinstance(x, torch.Tensor):
            return x.float()
        else:
            raise ValueError("Unknown input array type: {}".format(type(x)))

"""Converts a given numpy array to torch.FloatTensor."""
class ToFloatNumpy(object):

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            return x.astype("float32")
        elif isinstance(x, torch.Tensor):
            return x.float().numpy()
        else:
            raise ValueError("Unknown input array type: {}".format(type(x)))

"""Pre-Emphasize in order to raise higher frequencies and lower low frequencies."""
# high frequencies are more likely to be dampened during transmission
# thus have less intensity, compared to the lower frequency parts, e.g. F0, 1st harmonic, 2nd harmonic, etc..
class PreEmphasize(object):
    def __init__(self, factor=0.97):
        self.factor = factor

    def __call__(self, y):
        if y.dim() != 2:
            raise ValueError(
                "PreEmphasize expects a 2 dimensional signal of size (c, n), "
                "but got size: {}.".format(y.size())
            )
        return torch.cat(
            (y[:, 0].unsqueeze(dim=-1), y[:, 1:] - self.factor * y[:, :-1]), dim=-1 # purpose: remove sharpe edges
        )

# spectrogram related starts here
"""Converts a given audio to a spectrogram (amplitude not log scaled)."""
class Spectrogram(object):


    def __init__(self, n_fft, hop_length, center=True):
        # n_fft: resolution on freq axis
        self.n_fft = n_fft
        self.hop_length = hop_length # resolution on time axis; width of hann window/4; overlap shd add up to 1
        # center - true: t-th frame in spectrogram is centered at time t x hop_length of the signal
        # --> create 1-to-1 correspondence
        # done by reflected padding (padding on both sides)
        self.center = center
        # hann window: more weight on 'current' freqs at time t (weighting functions/weight matrix used in FFT analysis)
        # window functions control the amount of signal leakage between freq bins of FFT
        self.window = torch.hann_window(self.n_fft)

    def __call__(self, y):
        if y.dim() != 2:
            raise ValueError(
                "Spectrogram expects a 2 dimensional signal of size (c, n), "
                "but got size: {}.".format(y.size())
            )
        # Fourier transform: 2 inputs and 2 output dimensions (re, im).
        # For each frequency bin, the magnitude sqrt(re^2 + im^2) tells you the amplitude of the component at the corresponding frequency.
        # amplitude/color in spectrogram: height in each FT
        # The phase atan2(im, re) tells you the relative phase of that component.
        # STFT sliding is similar to sliding windows of conv2D
        # n_frames = ((data_len - (win_size - 1) - 1) / hop_size) + 1
        # librosa stft returns: (1 + n_fft/2, n_frames) (comparison of librosa and torch stft: https://www.programmersought.com/article/76425845654/)
        # torch STFT returns(* x N x T x 2) --> N: freq, T: time_frames, last dim: real and imaginary components
        # rea = spec[:, :, 0]
        # imag = spec[:, :, 1]
        # use rea and imag to calc magnitude and phase
        S = torch.stft(
            input=y, # shape(1 x num_samples)
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=self.center, # orcaspot: center=False
            onesided=True,
        ).transpose(1, 2) # --> (1, T, 257, 2) or (* x T x N x 2); this is complex valued STFT
        S /= self.window.pow(2).sum().sqrt() # energy is lost due to windowing, scale factor to recover energy
        S = S.pow(2).sum(-1) # get power of "complex" tensor (--> real number) and sum along last dim, returns (1xTxN) or (1 t f)
        return S


"""
Converts a given audio to a spectrogram, cache and store the spectrograms.
called by audiodataset.py during training
"""
class CachedSpectrogram(object):
    version = 4

    def __init__(
        self, cache_dir, spec_transform, file_reader=None, file_writer=None, **meta
    ):
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        if file_reader is not None:
            self.reader = file_reader
        else:
            self.reader = AsyncFileReader(n_readers=1)
        self.transform = spec_transform
        self.meta = meta
        if file_writer is not None:
            self.writer = file_writer
        else:
            self.writer = AsyncFileWriter(write_fn=self._write_fn, n_writers=1)

    def get_cached_name(self, file_name):
        """Return the abs path of the cached spectrogram """
        cached_spec_n = os.path.splitext(os.path.basename(file_name))[0] + ".spec"
        dir_structure = os.path.dirname(file_name).replace(r"/", "_") + "_"  # why: r"/" not "/"
        cached_spec_n = dir_structure + cached_spec_n
        if not os.path.isabs(cached_spec_n):
            cached_spec_n = os.path.join(self.cache_dir, cached_spec_n)
        return cached_spec_n

    def __call__(self, fn):
        """ __call__ allows the class's instance to be called as a function form
         where .spec is computed and created
         """
        cached_spec_n = self.get_cached_name(fn)
        if not os.path.isfile(cached_spec_n):
            return self._compute_and_cache(fn)
        try:
            data = self.reader(cached_spec_n)
            # torch.load() allows to load tensors;
            # io.BytesIO(): data kept as bytes in an in-memory buffer
            spec_dict = torch.load(io.BytesIO(data), map_location="cpu")  # load the tensors to CPU
        except (EOFError, RuntimeError):
            return self._compute_and_cache(fn)
        # keys in spec_dict: "v", "data"
        if not (
            "v" in spec_dict
            and spec_dict["v"] == self.version
            and "data" in spec_dict
            and spec_dict["data"].dim() == 3
        ):
            return self._compute_and_cache(fn)
        for key, value in self.meta.items():
            if not (key in spec_dict and spec_dict[key] == value):
                return self._compute_and_cache(fn)
        return spec_dict["data"]

    def _compute_and_cache(self, fn):
        """ 1. use Asynchronous file reader to read in audio data
            2. keep the above result as bytes in cpu
            3. perform transform (i.e. PreEmphasize, stft)
            fn: file name
         """
        try:
            audio_data = self.reader(fn)
            spec = self.transform(io.BytesIO(audio_data)) # spectrogramerzeuger
        except Exception:
            spec = self.transform(fn)
        self.writer(self.get_cached_name(fn), spec) # spectrogramerzeuger
        return spec

    def _write_fn(self, fn, data):
        spec_dict = {"v": self.version, "data": data}
        for key, value in self.meta.items():
            spec_dict[key] = value
        torch.save(spec_dict, fn)

# min_max normalization (implemented in AnimalSpot) to take care of different recorders / distance of animals to the recorder
# shall be used to replace db Normalization

"""Normalize a spectrogram by subtracting mean and dividing by std."""
class MeanStdNormalize(object):

    def __call__(self, spectrogram, ret_dict=None):
        mean = spectrogram.mean() # like normalize a pic, substract mean of the whole spectrogram
        spectrogram.sub_(mean) # like normalize a pic, substract mean of the whole spectrogram
        std = spectrogram.std()
        spectrogram.div_(std)
        if ret_dict is not None:
            ret_dict["mean"] = mean
            ret_dict["std"] = std
        return spectrogram

"""Normalize db scale to 0..1"""
# first amplitude is converted to db scale (class Amp2Db) and then db normalization is applied
class Normalize(object):

    def __init__(self, min_level_db=-100, ref_level_db=20):
        """
            min_level_db: normalization range - limit it to -100db (anything below -100db will be set to -100)
            ref_level_db: reference level db, theoretically 20db is the sound of air. --> upper bound of loudness (determined by samples in the dataset)
        """
        self.min_level_db = min_level_db
        self.ref_level_db = ref_level_db

    def __call__(self, spec):
        """
            tf.clip_by_value
        """
        return torch.clamp(
            (spec - self.ref_level_db - self.min_level_db) / -self.min_level_db, 0, 1
        )

"""Turns a spectrogram from the power/amplitude scale to the decibel scale (log scale of amplitudes)."""
class Amp2Db(object):

    def __init__(self, min_level_db=None, stype="power"):
        self.stype = stype
        self.multiplier = 10. if stype == "power" else 20.
        if min_level_db is None:
            self.min_level = None
        else:
            min_level_db = -min_level_db if min_level_db > 0 else min_level_db
            self.min_level = torch.tensor(
                np.exp(min_level_db / self.multiplier * np.log(10))
            )

    def __call__(self, spec):
        if self.min_level is not None:
            spec_ = torch.max(spec, self.min_level)
        else:
            spec_ = spec
        spec_db = self.multiplier * torch.log10(spec_) # log scale: dB-scale
        return spec_db

"""
Compress a spectrogram using torch.log1p(spec * compression_factor).
Intuition: very high amplitudes become not too far away from the rest;
low amplitudes with subtle differences become more distinguishable
not called anywhere
"""
class SPECLOG1P(object):

    def __init__(self, compression_factor=1):
        self.compression_factor = compression_factor

    def __call__(self, spectrogram):
        return torch.log1p(spectrogram * self.compression_factor)

"""
Decompress a spectrogram using torch.log1p(spec * compression_factor). Inverse of SPECLOG1P
not called anywhere
"""
class SPECEXPM1(object):

    def __init__(self, decompression_factor=1):
        self.decompression_factor = decompression_factor

    def __call__(self, spectrogram):
        return torch.expm1(spectrogram) / self.decompression_factor

"""
Scaling spectrogram dimension (time/frequency) by a given factor.
Called by TimeStretch and PitchShift
"""
def _scale(spectrogram: torch.Tensor, shift_factor: float, dim: int):
    in_dim = spectrogram.dim()
    if in_dim < 3:
        raise ValueError(
            "Expected spectrogram with size (c t f) or (n c t f)"
            ", but got {}".format(spectrogram.size())
        )
    if in_dim == 3:
        spectrogram.unsqueeze_(dim=0) # F.interpolate(): needs (n c t f)
    size = list(spectrogram.shape)[2:]
    dim -= 1
    size[dim] = int(round(size[dim] * shift_factor)) # increase or decrease the amount of bins
    spectrogram = F.interpolate(spectrogram, size=size, mode="nearest") # scale with interpolation (next neighbor) to the new shape
    if in_dim == 3:
        spectrogram.squeeze_(dim=0) # reshaped to (c t f)
    return spectrogram

# Data Augmentation starts
"""
Randomly shifts the pitch of a spectrogram by a factor of 2**Uniform(log2(from), log2(to)) (small pitch shift, not drastic).
Human perceive pitch logarithmically, in log2 (https://www.reddit.com/r/musictheory/comments/360ner/why_do_humans_perceive_sound_in_log_base_2/)
PitchShift: change how fast the original cycle is being played back, without changing duration 
            the harmonic structure is still preserved (i.e. freq peaks are multiples of fundamental freq)
"""
class RandomPitchSift(object):

    def __init__(self, from_=0.5, to_=1.5):
        self.from_ = math.log2(from_) #log2(0.5) = -1
        self.to_ = math.log2(to_) #log2(0.5) = 0.58

    def __call__(self, spectrogram: torch.Tensor):
        # uniform distribution; number of steps through which pitch must be shifted
        factor = 2 ** torch.empty((1,)).uniform_(self.from_, self.to_).item() # .item() return value of this tensor as standard Python number
        median = spectrogram.median()
        size = list(spectrogram.shape)
        scaled = _scale(spectrogram, factor, dim=2)
        if factor > 1:
            out = scaled[:, :, : size[2]]
        else:
            out = torch.full(size, fill_value=median, dtype=spectrogram.dtype)
            new_f_bins = int(round(size[2] * factor))
            out[:, :, 0:new_f_bins] = scaled
        return out

"""Randomly stretches the time of a spectrogram by a factor of 2**Uniform(log2(from), log2(to))."""
class RandomTimeStretch(object):

    def __init__(self, from_=0.5, to_=2):
        self.from_ = math.log2(from_)
        self.to_ = math.log2(to_)

    def __call__(self, spectrogram: torch.Tensor):
        factor = 2 ** torch.empty((1,)).uniform_(self.from_, self.to_).item()
        return _scale(spectrogram, factor, dim=1)

"""
Randomly scaling (uniform distributed) the amplitude based on a given input spectrogram (intensity augmenation).
(color in spectrogram)
"""
class RandomAmplitude(object):
    def __init__(self, increase_db=3, decrease_db=None):
        self.inc_db = increase_db
        if decrease_db is None:
            decrease_db = -increase_db
        elif decrease_db > 0:
            decrease_db *= -1
        self.dec_db = decrease_db

    def __call__(self, spec):
        db_change = torch.randint(
            self.dec_db, self.inc_db, size=(1,), dtype=torch.float
        )
        return spec.mul(10 ** (db_change / 10))

""" 
Randomly adds a given noise file to the given spectrogram by considering a randomly selected
(uniform distributed) SNR of min = -3 dB and max = 12 dB. The noise file could also be intensity, pitch, and/or time
augmented. If a noise file is longer/shorter than the given spectrogram it will be subsampled/self-concatenated. 
The spectrogram is expected to be a power spectrogram, which is **not** logarithmically compressed (for calc SNR).
Perform also histogram equalizing so that it looks more like a spectrogram
"""
class RandomAddNoise(object):

    def __init__(
        self,
        noise_files: List[str],
        spectrogram_transform,
        transform,
        min_length=0,
        min_snr=12,
        max_snr=-3,
        return_original=False,
    ):
        if not noise_files:
            raise ValueError("No noise files found")
        self.noise_files = noise_files
        self.t_spectrogram = spectrogram_transform
        self.noise_file_locks = {file: Lock() for file in noise_files}
        self.transform = transform
        self.min_length = min_length
        self.t_pad = PaddedSubsequenceSampler(sequence_length=min_length, dim=1)
        self.min_snr = min_snr if min_snr > max_snr else max_snr
        self.max_snr = max_snr if min_snr > max_snr else min_snr
        self.return_original = return_original

    def __call__(self, spectrogram):
        if len(self.noise_files) == 1:
            idx = 0
        else:
            idx = torch.randint(
                0, len(self.noise_files) - 1, size=(1,), dtype=torch.long
            ).item()
        noise_file = self.noise_files[idx]

        try:
            if not self.noise_file_locks[noise_file].acquire(timeout=10):
                print("Warning: Could not acquire lock for {}".format(noise_file))
                return spectrogram
            noise_spec = self.t_spectrogram(noise_file)
        except Exception:
            import traceback

            print(traceback.format_exc())
            return spectrogram
        finally:
            self.noise_file_locks[noise_file].release()

        noise_spec = self.t_pad._maybe_sample_subsequence(
            noise_spec, spectrogram.size(1) * 2
        )
        noise_spec = self.transform(noise_spec)

        if self.min_length > 0:
            spectrogram = self.t_pad._maybe_pad(spectrogram)

        if spectrogram.size(1) > noise_spec.size(1):
            n_repeat = int(math.ceil(spectrogram.size(1) / noise_spec.size(1)))
            noise_spec = noise_spec.repeat(1, n_repeat, 1)
        if spectrogram.size(1) < noise_spec.size(1):
            high = noise_spec.size(1) - spectrogram.size(1)
            start = torch.randint(0, high, size=(1,), dtype=torch.long)
            end = start + spectrogram.size(1)
            noise_spec_part = noise_spec[:, start:end]
        else:
            noise_spec_part = noise_spec

        snr = torch.randint(self.max_snr, self.min_snr, size=(1,), dtype=torch.float)
        signal_power = spectrogram.sum()
        noise_power = noise_spec_part.sum()

        K = (signal_power / noise_power) * 10 ** (-snr / 10) # maintain a certain SNR after injecting noises
        spectrogram_aug = spectrogram + noise_spec_part * K

        if self.return_original:
            return spectrogram_aug, spectrogram
        return spectrogram_aug

"""
Samples a subsequence along one axis and pads if necessary.
used both in RandomAddNoise() and sample signal files
"""
class PaddedSubsequenceSampler(object):

    def __init__(self, sequence_length: int, dim: int = 0, random=True):
        assert isinstance(sequence_length, int)
        assert isinstance(dim, int)
        self.sequence_length = sequence_length
        self.dim = dim
        if random: # if using augmentation
            self._sampler = lambda x: torch.randint(
                0, x, size=(1,), dtype=torch.long
            ).item()
        else:
            self._sampler = lambda x: x // 2

    def _maybe_sample_subsequence(self, spectrogram, sequence_length=None):
        if sequence_length is None:
            sequence_length = self.sequence_length
        sample_length = spectrogram.shape[self.dim]
        if sample_length > sequence_length:
            start = self._sampler(sample_length - sequence_length)
            end = start + sequence_length
            indices = torch.arange(start, end, dtype=torch.long)
            # torch.index_select(): returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor
            return torch.index_select(spectrogram, self.dim, indices)
        return spectrogram

    def _maybe_pad(self, spectrogram, sequence_length=None):
        if sequence_length is None:
            sequence_length = self.sequence_length
        sample_length = spectrogram.shape[self.dim]
        if sample_length < sequence_length:
            start = self._sampler(sequence_length - sample_length)
            end = start + sample_length

            shape = list(spectrogram.shape)
            shape[self.dim] = sequence_length
            padded_spectrogram = torch.zeros(shape, dtype=spectrogram.dtype)

            if self.dim == 0:
                padded_spectrogram[start:end] = spectrogram
            elif self.dim == 1:
                padded_spectrogram[:, start:end] = spectrogram
            elif self.dim == 2:
                padded_spectrogram[:, :, start:end] = spectrogram
            elif self.dim == 3:
                padded_spectrogram[:, :, :, start:end] = spectrogram
            return padded_spectrogram
        return spectrogram

    def __call__(self, spectrogram):
        spectrogram = self._maybe_pad(spectrogram)
        spectrogram = self._maybe_sample_subsequence(spectrogram)
        return spectrogram

"""
Frequency compression of a given frequency range into a chosen number of frequency bins.
used for frequency compression - linear
"""
class Interpolate(object):
    def __init__(self, n_freqs, sr=None, f_min=0, f_max=None):
        self.n_freqs = n_freqs # n_freq_bins --> defines the last dim of input to network
        self.sr = sr
        self.f_min = f_min
        self.f_max = f_max

    def __call__(self, spec):
        # num of freq bins = n_fft/2 + 1; f_fft: num of samples for computing each frame
        n_fft = (spec.size(2) - 1) * 2 # spec.size(2): get number of freq bins from spec

        if self.sr is not None and n_fft is not None:
            # freq resolution/bin width = sample_freq / n_fft (https://dsp.stackexchange.com/questions/31203/frequency-resolution-of-dft)
            min_bin = int(max(0, math.floor(n_fft * self.f_min / self.sr)))
            max_bin = int(min(n_fft - 1, math.ceil(n_fft * self.f_max / self.sr)))
            spec = spec[:, :, min_bin:max_bin] # input to network is focused between f_min and f_max

        spec.unsqueeze_(dim=0) # input shape for interpolate(): (mini-batch x channels x [optional depth] x [optional height] x width)
        # F.interpolate(): down/up samples the input to either the given size or the given scale_factor
        # don't understand how this interpolation work
        spec = F.interpolate(spec, size=(spec.size(2), self.n_freqs), mode="nearest") # size=(time_frames, self.n_freqs);
        return spec.squeeze(dim=0)

# Hz <-> Mel
"""Convert hertz to mel."""
def _hz2mel(f):
    return 2595 * np.log10(1 + f / 700)

"""Convert mel to hertz."""
def _mel2hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)

"""
Create melbank.
Code from https://github.com/pytorch/audio/blob/5787787edc/torchaudio/transforms.py
Access Data: 12.09.2018, Last Access Date: 08.12.2019
"""
def _melbank(sr, n_fft, n_mels=128, f_min=0.0, f_max=None, inverse=False):
    m_min = 0. if f_min == 0 else _hz2mel(f_min)
    m_max = _hz2mel(f_max if f_max is not None else sr // 2)

    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel2hz(m_pts)

    bins = torch.floor(((n_fft - 1) * 2 + 1) * f_pts / sr).long()

    fb = torch.zeros(n_mels, n_fft)
    for m in range(1, n_mels + 1):
        f_m_minus = bins[m - 1].item()
        f_m = bins[m].item()
        f_m_plus = bins[m + 1].item()

        if f_m_minus != f_m:
            fb[m - 1, f_m_minus:f_m] = (torch.arange(f_m_minus, f_m) - f_m_minus).float() / (
                f_m - f_m_minus
            )
        if f_m != f_m_plus:
            fb[m - 1, f_m:f_m_plus] = (f_m_plus - torch.arange(f_m, f_m_plus)).float() / (
                f_m_plus - f_m
            )

    if not inverse:
        return fb.t()
    else:
        return fb


"""
This turns a normal STFT into a MEL Frequency STFT, using a conversion matrix.  This uses triangular filter banks.
Code from https://github.com/pytorch/audio/blob/5787787edc/torchaudio/transforms.py
Access Data: 12.09.2018, Last Access Date: 08.12.2019
"""
class F2M(object):


    def __init__(
        self, sr: int = 16000, n_mels: int = 40, f_min: int = 0, f_max: int = None
    ):
        self.sr = sr
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sr // 2
        if self.f_max > self.sr // 2:
            raise ValueError("f_max > sr // 2")

    def __call__(self, spec_f: torch.Tensor):
        n_fft = spec_f.size(2)

        fb = _melbank(self.sr, n_fft, self.n_mels, self.f_min, self.f_max)

        spec_m = torch.matmul(
            spec_f, fb
        )
        return spec_m


"""
Converts a normal STFT into a MEL Frequency STFT, using a conversion
matrix. This uses triangular filter banks.
"""
class M2F(object):

    def __init__(
        self, sr: int = 16000, n_fft: int = 1024, f_min: int = 0, f_max: int = None
    ):
        self.sr = sr
        self.n_fft = n_fft // 2 + 1
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sr // 2
        if self.f_max > self.sr // 2:
            raise ValueError("f_max > sr // 2")

    def __call__(self, spec_m: torch.Tensor):
        n_mels = spec_m.size(2)

        fb = _melbank(self.sr, self.n_fft, n_mels, self.f_min, self.f_max, inverse=True)

        spec_f = torch.matmul(
            spec_m, fb
        )
        return spec_f

"""
Converts MEL Frequency to MFCC.
"""
class M2MFCC(object):
    def __call__(self, spec_m):
        device = spec_m.device
        if isinstance(spec_m, torch.Tensor):
            spec_m = spec_m.cpu().numpy()
        mfcc = scipy.fftpack.dct(spec_m, axis=-1)
        return torch.from_numpy(mfcc).to(device)

#if __name__ == "__main__":
