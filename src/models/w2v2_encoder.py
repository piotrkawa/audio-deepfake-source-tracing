"""
DISCLAIMER:
This code is provided "as-is" without any warranty of any kind, either expressed or implied,
including but not limited to the implied warranties of merchantability and fitness for a particular purpose.
The author assumes no liability for any damages or consequences resulting from the use of this code.
Use it at your own risk.
"""

import torch
from transformers import AutoFeatureExtractor, AutoModelForCTC


class Wav2Vec2Encoder(torch.nn.Module):
    def __init__(self, device: str = "cuda", sr: int = 16_000):
        super().__init__()
        self.sr = sr
        self.model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(
            device
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base-960h", sampling_rate=sr
        )
        self.LL = torch.nn.Sequential(
            torch.nn.Linear(768, 256), torch.nn.ReLU(), torch.nn.Linear(256, 256)
        ).to(device)

    def forward(self, x):
        input_values = self.feature_extractor(
            x, return_tensors="pt", sampling_rate=self.sr
        ).input_values
        input_values = input_values.to(self.model.device).squeeze()
        outputs = self.model(input_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        hidden_states = hidden_states.mean(1) # B x T x D -> B x D
        return self.LL(hidden_states)
