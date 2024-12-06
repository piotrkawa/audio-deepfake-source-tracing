import glob
import os
import random

import librosa
import numpy
import torch
import transformers
from scipy import signal


class HuggingFaceFeatureExtractor:
    def __init__(self, model_class_name, layer=-1, name=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(name)
        model_class = getattr(transformers, model_class_name)

        self.model = model_class.from_pretrained(name, output_hidden_states=True)
        self.model.eval()
        self.model.to(self.device)
        self.layer = layer

    def __call__(self, audio, sr):
        inputs = self.feature_extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.hidden_states[self.layer]


class WaveformEmphasiser:
    def __init__(self, sampling_rate, musan_path, rir_path):
        self.sampling_rate = sampling_rate
        self.noisesnr = {"noise": [0, 15], "speech": [13, 20], "music": [5, 15]}
        self.numnoise = {"noise": [1, 1], "speech": [3, 8], "music": [1, 1]}
        self.noiselist = {}
        self.rir_files = glob.glob(os.path.join(rir_path, "*/*/*/*.wav"))

        self.augment_files = glob.glob(os.path.join(musan_path, "*/*/*.wav"))
        ## group the noises by category
        for file in self.augment_files:
            if file.split("/")[-3] not in self.noiselist:
                self.noiselist[file.split("/")[-3]] = []
            self.noiselist[file.split("/")[-3]].append(file)

    def __call__(self, waveform, emphasis="original"):
        waveform = self._unpack(waveform)
        if emphasis == "original":
            waveform = waveform
        elif emphasis == "reverb":
            waveform = self.add_reverb(waveform)
        elif emphasis in ["speech", "music", "noise"]:
            waveform = self.add_noise(waveform, "speech")

        return self._pack(waveform)

    def _unpack(self, waveform):
        return waveform.squeeze().cpu().numpy()

    def _pack(self, waveform):
        return torch.Tensor(waveform)

    def add_reverb(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, sr = librosa.load(rir_file, sr=self.sampling_rate)
        rir = rir / numpy.sqrt(numpy.sum(rir**2))
        # print(f"Audio shape: {audio.shape}")
        # print(f"RIR shape: {rir.shape}")
        result = signal.convolve(audio, rir, mode="full")[: audio.shape[0]]
        return result

    def add_noise(self, audio, noise_type="speech"):
        audio_db = 10 * numpy.log10(numpy.mean(audio**2) + 1e-4)
        noise_file = random.choice(self.noiselist[noise_type])
        noise, sr = librosa.load(noise_file, sr=self.sampling_rate)
        if noise.shape[0] <= audio.shape[0]:
            noise = numpy.pad(noise, (0, audio.shape[0] - noise.shape[0]), "wrap")
        else:
            noise = noise[: audio.shape[0]]
        noise_db = 10 * numpy.log10(numpy.mean(noise**2) + 1e-4)
        random_noise_snr = random.uniform(
            self.noisesnr[noise_type][0], self.noisesnr[noise_type][1]
        )
        noise = (
            numpy.sqrt(10 ** ((audio_db - noise_db - random_noise_snr) / 10)) * noise
        )
        result = audio + noise
        return result


def shuffle(
    feat: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    shuffle_index = torch.randperm(labels.shape[0])
    feat = feat[shuffle_index]
    labels = labels[shuffle_index]
    return feat, labels
