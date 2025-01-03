import torch
from torch import nn

from src.models.w2v2_encoder import Wav2Vec2Encoder


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return total, trainable, non_trainable


def get_model(model_name: str, checkpoint_path: str | None) -> nn.Module:
    if model_name == "wav2vec2":
        model = Wav2Vec2Encoder()
    else:
        raise ValueError(f"Model '{model_name}' not implemented")

    print(f" > Initialized model '{model_name}'")

    total, trainable, non_trainable = count_parameters(model)
    print(
        f" > Number of parameters: {total:,}, Trainable: {trainable:,}, Non-Trainable: {non_trainable:,}"
    )

    if checkpoint_path:
        print(f" > Loading weights from '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])

    return model
