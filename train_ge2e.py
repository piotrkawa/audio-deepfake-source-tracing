"""
DISCLAIMER:
This code is provided "as-is" without any warranty of any kind, either expressed or implied,
including but not limited to the implied warranties of merchantability and fitness for a particular purpose.
The author assumes no liability for any damages or consequences resulting from the use of this code.
Use it at your own risk.

## Author: Piotr KAWA
## December 2024
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim
import yaml
from sklearn.calibration import LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import samplers
from src.datasets.dataset import BaseDataset
from src.losses import GE2ELoss
from src.models import get_model
from src.utils import set_seed


def reshape_to_loss_format(
    x: torch.Tensor,
    num_utter_per_class: int,
    num_classes_in_batch: int,
) -> torch.Tensor:
    return (
        x.contiguous()
        .reshape(num_utter_per_class, num_classes_in_batch, x.shape[-1])
        .transpose(0, 1)
        .contiguous()
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force the use of CPU even if GPU is available",
    )
    return parser.parse_args()


def initialize_datasets(
    config: dict, path_mlaad: Path, path_protocols: Path, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    protocols = {
        "train": path_protocols / "train.csv",
        "dev": path_protocols / "dev.csv",
        "test": path_protocols / "eval.csv",
    }

    for subset, protocols_root_path in protocols.items():
        if not protocols_root_path.exists():
            raise FileNotFoundError(f"{protocols_root_path} does not exist")

    dataframes = {
        subset: pd.read_csv(protocols_root_path)
        for subset, protocols_root_path in protocols.items()
    }

    all_df = pd.concat(dataframes.values())
    le = LabelEncoder()
    le.fit(all_df["model_name"])
    class_mapping = {name: idx for idx, name in enumerate(le.classes_)}

    for subset, df in dataframes.items():
        df["model_id"] = le.transform(df["model_name"])
        dataframes[subset] = df

    # we train on concatenation of train and dev
    train_and_dev = pd.concat([dataframes["train"], dataframes["dev"]])
    train_dataset = BaseDataset(
        meta_data=train_and_dev.to_dict(orient="records"),
        basepath=path_mlaad,
        class_mapping=class_mapping,
        sr=config["data"]["sampling_rate"],
        sample_length_s=config["data"]["sample_length_s"],
        verbose=True,
    )

    test_dataset = BaseDataset(
        meta_data=dataframes["test"].to_dict(orient="records"),
        basepath=path_mlaad,
        class_mapping=class_mapping,
        sr=config["data"]["sampling_rate"],
        sample_length_s=config["data"]["sample_length_s"],
        verbose=True,
    )

    train_sampler = samplers.PerfectBatchSampler(
        dataset_items=train_dataset.samples,
        classes=train_dataset.get_class_list(),
        batch_size=batch_size,
        num_classes_in_batch=n_classes_in_batch,
        num_gpus=1,
        drop_last=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config["data"]["num_workers"],
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        sampler=train_sampler,
    )

    test_sampler = samplers.PerfectBatchSampler(
        dataset_items=test_dataset.samples,
        classes=test_dataset.get_class_list(),
        batch_size=batch_size,
        num_classes_in_batch=n_classes_in_batch,
        num_gpus=1,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=test_dataset.collate_fn,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        sampler=test_sampler,
        drop_last=True,
    )

    return train_loader, test_loader


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    num_epochs: int,
    log_interval: int,
    save_path: Path,
    n_utter_per_class: int,
    n_classes_in_batch: int,
) -> torch.nn.Module:

    best_loss = float("inf")
    for epoch in tqdm(range(num_epochs)):
        tqdm.write(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        num_total = 0

        for batch_idx, (x, y, paths) in enumerate(train_loader):
            batch_size = y.size(0)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            output = model(x)
            out_reshaped = reshape_to_loss_format(
                output, n_utter_per_class, n_classes_in_batch
            )
            train_loss = criterion(out_reshaped)

            train_loss.backward()

            optimizer.step()
            running_loss += train_loss.item() * batch_size
            num_total += batch_size

            if (batch_idx + 1) % log_interval == 0:
                print(
                    f"Batch [{batch_idx+1}]: Train Loss: {running_loss / num_total:.4f}"
                )
        running_loss /= num_total

        print(f"Epoch [{epoch+1}/{num_epochs}]: Train Loss: {running_loss:.4f}")

        if running_loss < best_loss:
            model_save_path = save_path / "best_model.pth"
            print(
                f"Loss improved ({best_loss:.4f} -> {running_loss:.4f}). Saving model to '{model_save_path}'."
            )
            best_loss = running_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": running_loss,
                },
                model_save_path,
            )
    return model


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["training"]["seed"])

    path_mlaad = config["data"]["mlaad_root_path"]
    path_protocols = config["data"]["protocols_root_path"]

    model = get_model(
        model_name=config["model"]["model_name"],
        checkpoint_path=config["model"]["checkpoint_path"],
    )
    model = model.train()
    lr = config["training"]["lr"]
    num_epochs = config["training"]["num_epochs"]
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    num_workers = config["data"]["num_workers"]

    criterion = GE2ELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    path_mlaad = config["data"]["mlaad_root_path"]
    path_protocols = Path(config["data"]["protocols_root_path"])

    save_path = Path(config["training"]["save_path"])
    print(f"Model-related data will be saved in '{save_path}'")

    save_path.mkdir(parents=True, exist_ok=True)
    log_interval = config["training"]["log_interval"]
    n_classes_in_batch = config["training"]["n_classes_in_batch"]
    n_utter_per_class = config["training"]["n_utter_per_class"]

    batch_size = n_classes_in_batch * n_utter_per_class

    train_loader, test_loader = initialize_datasets(
        config=config,
        path_mlaad=path_mlaad,
        path_protocols=path_protocols,
        batch_size=batch_size,
    )

    model.to(device)
    model = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        log_interval=log_interval,
        save_path=save_path,
        n_classes_in_batch=n_classes_in_batch,
        n_utter_per_class=n_utter_per_class,
    )

    print("Finished training. Started test procedure!")
    model.eval()
    test_running_loss = 0
    num_total = 0

    with torch.no_grad():
        for batch_idx, (x, y, paths) in enumerate(test_loader):
            batch_size = y.size(0)
            x, y = x.to(device), y.to(device)

            output = model(x)
            out_reshaped = reshape_to_loss_format(
                output, n_utter_per_class, n_classes_in_batch
            )
            loss = criterion(out_reshaped)

            test_running_loss += loss.item() * batch_size
            num_total += batch_size

            if (batch_idx + 1) % log_interval == 0:
                print(
                    f"Batch [{batch_idx+1}]: Test Loss: {test_running_loss / num_total:.4f}"
                )

    test_running_loss /= num_total
    print(f"Test Loss: {test_running_loss:.4f}")
