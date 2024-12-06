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
from sklearn.metrics import classification_report, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.w2v2_aasist import W2VAASIST
from src.datasets.dataset import MLAADFDDataset
from src.models.NSD import NSDOODDetector


def parse_args():
    parser = argparse.ArgumentParser("OOD Detector script")
    # Paths
    parser.add_argument(
        "--model_path",
        type=str,
        default="exp/trained_models/anti-spoofing_feat_model.pt",
        help="Path to trained model",
    )
    parser.add_argument(
        "--feature_path",
        type=str,
        default="exp/preprocess_wav2vec2-base/",
        help="Path to features",
    )
    parser.add_argument(
        "--out_folder", type=str, default="exp/ood_step/", help="Path to output results"
    )
    parser.add_argument(
        "--label_assignment_path",
        type=str,
        default="exp/label_assignment.txt",
        help="Path to the file which lists the class assignments as written in the preprocessing step",
    )
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=128, help="Batch_size")
    parser.add_argument("--feat_dim", type=int, default=768, help="Feature dimension")
    parser.add_argument("--hidden_dim", type=int, default=160, help="Hidden size dim")
    parser.add_argument(
        "--num_classes", type=int, default=24, help="Number of known systems"
    )
    parser.add_argument(
        "--feature_extraction_step",
        action="store_true",
        help="Whether to run the feature extraction step or just the OOD",
    )

    args = parser.parse_args()
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    return args


def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    eer_index = np.nanargmin(np.abs(fpr - (1 - tpr)))
    return fpr[eer_index], thresholds[eer_index]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.model_path}")
    model = W2VAASIST(args.feat_dim, args.num_classes)
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if args.feature_extraction_step:
        # Loading datasets
        train_dataset = MLAADFDDataset(args.feature_path, "train")
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

        dev_dataset = MLAADFDDataset(args.feature_path, "dev")
        dev_loader = DataLoader(
            dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

        eval_dataset = MLAADFDDataset(args.feature_path, "eval")
        eval_loader = DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

        # Extract logits and hidden states from trained model
        for subset_, loader in zip(
            ["train", "dev", "eval"], [train_loader, dev_loader, eval_loader]
        ):
            print(f"Running hidden feature extraction for {subset_}")
            all_feats = np.zeros((len(loader) * args.batch_size, args.hidden_dim))
            all_logits = np.zeros((len(loader) * args.batch_size, args.num_classes))
            all_labels = np.zeros(len(loader) * args.batch_size)

            for idx, batch in enumerate(tqdm(loader)):
                sample_num = idx * args.batch_size
                feats, filename, labels = batch
                feats = feats.transpose(2, 3).to(device)
                with torch.no_grad():
                    hidden_state, logits = model(feats)
                # Store all info
                all_feats[sample_num : sample_num + feats.shape[0]] = (
                    hidden_state.detach().cpu().numpy()
                )
                all_logits[sample_num : sample_num + feats.shape[0]] = (
                    logits.detach().cpu().numpy()
                )
                all_labels[sample_num : sample_num + feats.shape[0]] = labels
            # Save the info
            out_path = os.path.join(args.out_folder, f"{subset_}_dict.npy")
            np.save(
                out_path,
                {"feats": all_feats, "logits": all_logits, "labels": all_labels},
            )
            print(f"Saved hidden_states to {out_path}")

    train_dict = np.load(
        os.path.join(args.out_folder, "train_dict.npy"), allow_pickle=True
    ).item()
    dev_dict = np.load(
        os.path.join(args.out_folder, "dev_dict.npy"), allow_pickle=True
    ).item()
    eval_dict = np.load(
        os.path.join(args.out_folder, "eval_dict.npy"), allow_pickle=True
    ).item()

    print("Setting up the OOD detector using the training data...")
    ood_detector = NSDOODDetector()
    ood_detector.setup(args, train_dict)

    # Get scores for OOD
    print("Getting OOD scores for the dev set...")
    dev_scores = ood_detector.infer(dev_dict)

    # Get the systems' labels assigned to OOD samples
    # Convert the system numbers into classes: OOD=1 and KNOWN=0
    with open(args.label_assignment_path) as f:
        OOD_classes = [
            int(line.split("|")[1])
            for line in f.readlines()
            if line.strip().split("|")[2] == "OOD"
        ]
    dev_ood_labels = [
        1 if int(dev_dict["labels"][k]) in OOD_classes else 0
        for k in range(len(dev_dict["labels"]))
    ]

    # Compute a EER threshold over the dev scores
    print("\nComputing the EER threshold over the development set...")
    eer, threshold = compute_eer(dev_ood_labels, dev_scores)
    print(f"DEV EER: {eer*100:.2f}  | Threshold: {threshold:.2f}")

    # Set the threshold and compute the OOD accuracy over the eval set
    print("\nComputing the evaluation results using the dev threshold...")
    print("Class 1 is OOD, Class 0 is ID")
    eval_scores = ood_detector.infer(eval_dict)
    eval_ood_labels = [
        1 if int(eval_dict["labels"][k]) in OOD_classes else 0
        for k in range(len(eval_dict["labels"]))
    ]
    predicts = [
        1 if eval_scores[k] > threshold else 0 for k in range(len(eval_dict["labels"]))
    ]

    print("OOD classification report for eval data:")
    report = classification_report(eval_ood_labels, predicts)
    report_path = os.path.join(args.out_folder, "OOD_eval_results.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(report)
    print(f"... also written to {report_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
