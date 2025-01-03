"""
DISCLAIMER:
This code is provided "as-is" without any warranty of any kind, either expressed or implied,
including but not limited to the implied warranties of merchantability and fitness for a particular purpose.
The author assumes no liability for any damages or consequences resulting from the use of this code.
Use it at your own risk.

### Adapted from: https://github.com/xieyuankun/REFD/blob/main/code/ADD2023t3_FD/ood_detectors/NSD.py
### and: https://github.com/xieyuankun/REFD/blob/main/code/ADD2023t3_FD/ood_detectors/interface.py


## Author: Adriana STAN
## December 2024
## Parts of this code are taken from https://github.com/xieyuankun/REFD/tree/main/code/ADD2023t3_FD
"""

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


class OODDetector(ABC):
    @abstractmethod
    def setup(self, args, train_model_outputs: Dict[str, torch.Tensor]):
        pass

    @abstractmethod
    def infer(self, model_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass


def NSD_with_angle(feats_train, feats, min=False):
    feas_train = feats_train.cpu().numpy()
    feats = feats.cpu().numpy()
    cos_similarity = np.dot(feats, feats_train.T)
    if min:
        scores = np.array(cos_similarity.min(axis=1))
    else:
        scores = np.array(cos_similarity.mean(axis=1))
    return scores


class NSDOODDetector(OODDetector):
    def setup(self, args, train_model_outputs):
        # Compute the training set info
        logits_train = torch.Tensor(train_model_outputs["logits"])
        feats_train = torch.Tensor(train_model_outputs["feats"])
        train_labels = train_model_outputs["labels"]
        feats_train = F.normalize(feats_train, p=2, dim=-1)
        confs_train = torch.logsumexp(logits_train, dim=1)
        self.scaled_feats_train = feats_train * confs_train[:, None]

    def infer(self, model_outputs):
        feats = torch.Tensor(model_outputs["feats"])
        logits = torch.Tensor(model_outputs["logits"])
        feats = F.normalize(feats, p=2, dim=-1)
        confs = torch.logsumexp(logits, dim=1)
        guidances = NSD_with_angle(self.scaled_feats_train, feats)
        scores = torch.from_numpy(guidances).to(confs.device) * confs
        return scores
