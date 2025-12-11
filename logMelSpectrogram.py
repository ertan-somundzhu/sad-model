
import torch
import numpy as np
from matplotlib import pyplot as plt

from typing import Optional

class logMelSpectrogram:

    def __init__(
            self,
            frame_rate_s: int = 30,
            stride_s: int = 10,
            n_fft: Optional[int] = None,
            n_mels: Optional[int] = 40,
            top_db: int = 80,
            pre_emph_coef: float = 0.95,
            device: Optional[str] = None
    ):

        self.frame_rate_s = frame_rate_s
        self.stride_s = stride_s
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.log_mel_spec_is_computed = False
        self.top_db = top_db
        self.pre_emph_coef = pre_emph_coef

        if not device:
            self.device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.mps.is_available() else "cpu"
            )
        self.device = device
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float32)

    def transform(
            self,
            samples: np.array,
            sr: int,
    ):

        self.samples = torch.from_numpy(samples)
        self.sr = sr

        if self.samples.shape[0] < 2:
            raise ValueError("Samples should be longer than two")


        # pre emphasis
        # it's necessary to compensate the audio roll off
        # meaning it amplifies the difference between current signal
        # and previous one

        pre_emph_samples = torch.cat([
            self.samples[0:1],
            self.samples[1:] - self.pre_emph_coef * self.samples[:-1]
        ], dim=0)

        # framing
        # it's needed to turn the audio into descrete overlapping chunks

        stride = self.sr * self.stride_s // 1000
        frame_rate = self.sr * self.frame_rate_s // 1000


        chunks = pre_emph_samples.unfold(0, frame_rate, stride).contiguous()
        num_of_frames = chunks.shape[0]

        # hann window to smooth out the edges
        # as i understand, it is necessary to
        # smooth out the edges of chunks to avoid
        # sudden drops and rises in volume

        n = torch.arange(frame_rate)
        hanning_weights = 0.5 - 0.5 * torch.cos(2 * torch.pi * n / (frame_rate - 1))

        weighted_chunks = chunks * hanning_weights


        # applying fast fourier transform
        # to decompose "raw" audio into underlying frequencies
        # only positive frequencies are taken, because negative freqs
        # dont bring new information
        # so there are about "half" (n_fft / 2 + 1) extracted
        if not self.n_fft:
            self.n_fft = 2 ** torch.ceil(torch.log2(torch.tensor(frame_rate, dtype=torch.float32))).to(torch.int32)

        fft_chunks = torch.fft.rfft(weighted_chunks, n=self.n_fft)
        power_spec = (2 / self.n_fft ** 2) * torch.abs(fft_chunks) ** 2


        # herz to mels converter and vice versa

        def hz_to_mel(hz):
            return 2595 * torch.log10(1 + hz / 700)
        def mel_to_hz(m):
            return 700 * (10 ** (m / 2595) - 1)

        fmax = self.sr / 2
        fmin = 0

        # here we create mels scale
        mels = torch.linspace(
            hz_to_mel(torch.tensor(fmin)),
            hz_to_mel(torch.tensor(fmax)),
            self.n_mels + 2
        )

        # converting linear mels to hz thus
        # introducing non-linearity
        hz_points = mel_to_hz(mels)
        bins = torch.floor((self.n_fft + 1) * hz_points / self.sr).to(torch.int32)

        # building triangular filters
        # that are overlapping and gain "energy" with the increase of hz
        # simulating human hearing that is better at distinguishing between lower
        # freqs than higher ones
        # so as the hz rises the filter becomes bigger
        # and, if one might say, less sensitive
        k = torch.arange(self.n_fft // 2 + 1).unsqueeze(0)

        f_left = bins[:-2].unsqueeze(1)
        f_center = bins[1:-1].unsqueeze(1)
        f_right = bins[2:].unsqueeze(1)

        up = (k - f_left) / torch.clamp(f_center - f_left, min=1e-8)      # (n_mels, bins)
        down = (f_right - k) / torch.clamp(f_right - f_center, min=1e-8) # (n_mels, bins)

        filters = torch.clamp(torch.minimum(up, down), min=0.0)


        mel_spec = torch.matmul(filters, power_spec.T)

        # converting mel spectogram to log scale

        mel_spec = torch.clamp(mel_spec, min=1e-10)
        log_mel_spec = 10 * torch.log10(mel_spec)

        # normalising

        log_mel_spec = torch.clamp(
            log_mel_spec,
            min=torch.max(log_mel_spec) - self.top_db
        )

        self.log_mel_spec = log_mel_spec

        self.log_mel_spec_is_computed = True

        return log_mel_spec

    def plot_waveform(self):

        plt.figure(figsize=(10, 4))
        cpu_samples = self.samples.cpu().numpy()
        plt.plot(np.arange(cpu_samples.shape[0]) / self.sr, cpu_samples)
        plt.title("Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.show()

    def plot_log_mel_spec(self, cmap="magma_r"):

        if not self.log_mel_spec_is_computed:
            raise ValueError("run compute() before plotting log mel spectogram")

        plt.figure(figsize=(10, 4))
        spec_to_plot = self.log_mel_spec.cpu().numpy()
        plt.imshow(spec_to_plot, origin="lower", aspect="auto", cmap=cmap)
        plt.title("Log-Mel Spectrogram (dB)")
        plt.xlabel("Time frames")
        plt.ylabel("Mel bins")
        plt.colorbar()
        plt.show()
