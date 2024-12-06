"""
DISCLAIMER:
This code is provided "as-is" without any warranty of any kind, either expressed or implied,
including but not limited to the implied warranties of merchantability and fitness for a particular purpose.
The author assumes no liability for any damages or consequences resulting from the use of this code.
Use it at your own risk.

## Author: Adriana STAN
## December 2024
"""

import argparse
import os
import sys
from pathlib import Path

# Enables running the script from root directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.dataset import MLAADFDDataset
from src.models.w2v2_aasist import W2VAASIST


def parse_args():
    parser = argparse.ArgumentParser(description="Get metrics")
    parser.add_argument(
        "--model_path",
        type=str,
        default="exp/trained_models/anti-spoofing_feat_model.pt",
        help="Path to trained model",
    )
    parser.add_argument(
        "--path_to_features",
        type=str,
        default="exp/preprocess_wav2vec2-base/",
        help="Path to features",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="exp/results/",
        help="Where to write the results",
    )

    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for inference"
    )
    parser.add_argument(
        "--feat_dim",
        type=int,
        default=768,
        help="Feature dimension of wav2vec features",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=24,
        help="Number of systems in the training dataset",
    )
    args = parser.parse_args()
    if not os.path.exists((args.results_path)):
        os.makedirs(args.results_path)
    return args


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}..")

    # Read the data
    dev_dataset = MLAADFDDataset(
        args.path_to_features, "dev", mode="known", max_samples=-1
    )
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, num_workers=0)

    eval_dataset = MLAADFDDataset(
        args.path_to_features, "eval", mode="known", max_samples=-1
    )
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=0)

    if len(eval_dataset) == 0:
        print("No data found for evaluation! Exiting...")
        exit(1)

    print(f"Loading model from {args.model_path}")
    model = W2VAASIST(args.feat_dim, args.num_classes)
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("Running on dev data...")
    with torch.no_grad():
        all_predicted = np.zeros(len(dev_dataset), dtype=int)
        all_labels = np.zeros(len(dev_dataset), dtype=int)

        dev_bar = tqdm(dev_loader, desc=f"Evaluation")
        for idx, batch in enumerate(dev_bar):
            sample_number = idx * args.batch_size
            feats, filename, labels = batch
            feats = feats.transpose(2, 3).to(device)
            _, logits = model(feats)
            logits = F.softmax(logits, dim=1)
            predicted = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_predicted[sample_number : sample_number + labels.shape[0]] = predicted
            all_labels[sample_number : sample_number + labels.shape[0]] = labels

    print("Classification report for DEV data: ")
    report_path = os.path.join(args.results_path, "dev_in_domain_results.txt")
    report = classification_report(
        all_labels, all_predicted, labels=np.unique(all_labels), zero_division=1.0
    )
    with open(report_path, "w") as f:
        f.write(report)
    print(report)
    print(f"... also written to {report_path}")

    print("Running on evaluation data...")
    with torch.no_grad():
        all_predicted = np.zeros(len(eval_dataset), dtype=int)
        all_labels = np.zeros(len(eval_dataset), dtype=int)

        eval_bar = tqdm(eval_loader, desc=f"Evaluation")
        for idx, batch in enumerate(eval_bar):
            sample_number = idx * args.batch_size
            feats, filename, labels = batch
            feats = feats.transpose(2, 3).to(device)
            _, logits = model(feats)
            logits = F.softmax(logits, dim=1)
            predicted = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_predicted[sample_number : sample_number + labels.shape[0]] = predicted
            all_labels[sample_number : sample_number + labels.shape[0]] = labels

    print("Classification report for EVAL data:")
    report_path = os.path.join(args.results_path, "eval_in_domain_results.txt")
    report = classification_report(
        all_labels, all_predicted, labels=np.unique(all_labels), zero_division=1.0
    )
    with open(report_path, "w") as f:
        f.write(report)
    print(report)
    print(f"... also written to {report_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
