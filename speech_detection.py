

from pydub import AudioSegment
AudioSegment.converter = "/usr/bin/ffmpeg"

import numpy as np

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from typing import Optional

class detectSpeech:

    def __init__(
        self,
        model_class,
        logMelSpectrogram,
        model_path: str,
        stride_s: int = 25,
        frame_rate_s: int = 25,
        device: Optional[str] = None,
        threshold: float = 0.5,
        batch_size: int = 32,
        sr: int = 16000
    ):

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.mps.is_available() else "cpu"
            )
        else:
            self.device = device

        self.model_path = model_path

        self.model = model_class.to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
        self.model.eval()

        self.log_mel_spec = logMelSpectrogram

        self.sr = sr
        self.stride = sr * stride_s // 1000
        self.frame_rate = sr * frame_rate_s // 1000


    def detect(
        self,
        audio_path: str
    ):


        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(self.sr)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

        log_mel = self.log_mel_spec.transform(samples=samples, sr=self.sr).to(self.device)


        chunks_mel = log_mel.unfold(dimension=1, size=self.frame_rate, step=self.stride)
        chunks_mel = chunks_mel.permute(1, 0, 2)

        chunks_mel = F.normalize(chunks_mel).unsqueeze(1)

        with torch.no_grad():
            outputs = self.model.forward(chunks_mel)
            outputs = torch.sigmoid(outputs)
            outputs = (outputs >= 0.5).int()

            onset, offset = torch.split(outputs, 400, dim=1)


        return torch.flatten(onset), torch.flatten(offset)
