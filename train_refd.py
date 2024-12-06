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
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.sampler as torch_sampler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import utils
from src.datasets.dataset import MLAADFDDataset
from src.models.w2v2_aasist import W2VAASIST


def parse_args():
    parser = argparse.ArgumentParser("Training script parameters")

    # Paths to features and output
    parser.add_argument(
        "-f",
        "--path_to_features",
        type=str,
        default="./exp/preprocess_wav2vec2-base/",
        help="Path to the previuosly extracted features",
    )
    parser.add_argument(
        "--out_folder", type=str, default="./exp/trained_models/", help="Output folder"
    )

    # Training hyperparameters
    parser.add_argument("--seed", type=int, help="random number seed", default=688)
    parser.add_argument(
        "--feat_dim",
        type=int,
        default=768,
        help="Feature dimension from the wav2vec model",
    )
    parser.add_argument(
        "--num_classes", type=int, default=24, help="Number of in domain classes"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=30, help="Number of epochs for training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
    parser.add_argument(
        "--lr_decay", type=float, default=0.5, help="decay learning rate"
    )
    parser.add_argument("--interval", type=int, default=10, help="interval to decay lr")
    parser.add_argument("--beta_1", type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument("--beta_2", type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument("--eps", type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--num_workers", type=int, default=0, help="number of workers")
    parser.add_argument(
        "--base_loss",
        type=str,
        default="ce",
        choices=["ce", "bce"],
        help="Loss for basic training",
    )
    args = parser.parse_args()

    # Set seeds
    utils.set_seed(args.seed)

    # Path for output data
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    # Folder for intermediate results
    if not os.path.exists(os.path.join(args.out_folder, "checkpoint")):
        os.makedirs(os.path.join(args.out_folder, "checkpoint"))

    # Path for input data
    assert os.path.exists(args.path_to_features)

    # Save training arguments
    with open(os.path.join(args.out_folder, "args.json"), "w") as file:
        file.write(json.dumps(vars(args), sort_keys=True, separators=("\n", ":")))

    cuda = torch.cuda.is_available()
    print("Running on: ", "cuda" if cuda else "cpu")
    args.device = torch.device("cuda" if cuda else "cpu")
    return args


def train(args):

    # Load the train and dev data
    print("Loading training data...")
    training_set = MLAADFDDataset(args.path_to_features, "train")
    print("\nLoading dev data...")
    dev_set = MLAADFDDataset(args.path_to_features, "dev", mode="known")

    train_loader = DataLoader(
        training_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=torch_sampler.SubsetRandomSampler(range(len(training_set))),
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=torch_sampler.SubsetRandomSampler(range(len(dev_set))),
    )

    # Setup the model
    model = W2VAASIST(args.feat_dim, args.num_classes).to(args.device)
    print(f"Training a {type(model).__name__} model for {args.num_epochs} epochs")
    feat_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta_1, args.beta_2),
        eps=args.eps,
        weight_decay=0.0005,
    )
    if args.base_loss == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCELoss()

    prev_loss = 1e8
    # Main training loop
    for epoch_num in range(args.num_epochs):
        model.train()
        utils.adjust_learning_rate(args, args.lr, feat_optimizer, epoch_num)

        epoch_bar = tqdm(train_loader, desc=f"Epoch [{epoch_num+1}/{args.num_epochs}]")
        accuracy, train_loss = [], []
        for iter_num, batch in enumerate(epoch_bar):
            feat, audio, labels = batch
            feat = feat.transpose(2, 3).to(args.device)
            labels = labels.to(args.device)

            mix_feat, y_a, y_b, lam = utils.mixup_data(
                feat, labels, args.device, alpha=0.5
            )

            targets_a = torch.cat([labels, y_a])
            targets_b = torch.cat([labels, y_b])
            feat = torch.cat([feat, mix_feat], dim=0)

            feats, feat_outputs = model(feat)
            if args.base_loss == "bce":
                feat_loss = criterion(feat_outputs, labels.unsqueeze(1).float())
            else:
                feat_loss = utils.regmix_criterion(
                    criterion, feat_outputs, targets_a, targets_b, lam
                )

            score = F.softmax(feat_outputs, dim=1)  # [:, 0]
            predicted_classes = np.argmax(score.detach().cpu().numpy(), axis=1)
            correct_predictions = [
                1 for k in range(len(labels)) if predicted_classes[k] == labels[k]
            ]
            accuracy.append(sum(correct_predictions) / len(labels) * 100)
            train_loss.append(feat_loss.item())
            epoch_bar.set_postfix(
                {
                    "train_loss": f"{sum(train_loss)/(iter_num+1):.4f}",
                    "acc": f"{sum(accuracy)/(iter_num+1):.2f}",
                }
            )

            feat_optimizer.zero_grad()
            feat_loss.backward()
            feat_optimizer.step()

        # Epoch eval
        model.eval()
        with torch.no_grad():
            val_bar = tqdm(dev_loader, desc=f"Validation for epoch {epoch_num+1}")
            accuracy, val_loss = [], []
            for iter_num, batch in enumerate(val_bar):
                feat, _, labels = batch
                feat = feat.transpose(2, 3).to(args.device)
                labels = labels.to(args.device)

                feats, feat_outputs = model(feat)
                if args.base_loss == "bce":
                    feat_loss = criterion(feat_outputs, labels.unsqueeze(1).float())
                    score = feat_outputs
                else:
                    feat_loss = criterion(feat_outputs, labels)
                    score = F.softmax(feat_outputs, dim=1)

                predicted_classes = np.argmax(score.detach().cpu().numpy(), axis=1)
                correct_predictions = [
                    1 for k in range(len(labels)) if predicted_classes[k] == labels[k]
                ]
                accuracy.append(sum(correct_predictions) / len(labels) * 100)

                val_loss.append(feat_loss.item())
                val_bar.set_postfix(
                    {
                        "val_loss": f"{sum(val_loss)/(iter_num+1):.4f}",
                        "val_acc": f"{sum(accuracy)/(iter_num+1):.2f}",
                    }
                )

        epoch_val_loss = sum(val_loss) / (iter_num + 1)
        if epoch_val_loss < prev_loss:
            # Save the checkpoint with better val_loss
            checkpoint_path = os.path.join(
                args.out_folder, "anti-spoofing_feat_model.pth"
            )
            print(f"[INFO] Saving model with better val_loss to {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)
            prev_loss = epoch_val_loss

        elif (epoch_num + 1) % 10 == 0:
            # Save the intermediate checkpoints just in case
            checkpoint_path = os.path.join(
                args.out_folder,
                "checkpoint",
                "anti-spoofing_feat_model_%02d.pth" % (epoch_num + 1),
            )
            print(
                f"[INFO] Saving intermediate model at epoch {epoch_num+1} to {checkpoint_path}"
            )
            torch.save(model.state_dict(), checkpoint_path)
        print("\n")


if __name__ == "__main__":
    args = parse_args()
    train(args)
