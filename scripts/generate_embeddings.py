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
import sys
from pathlib import Path

# Enables running the script from root directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import models
from src.datasets.dataset import BaseDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Generate embeddings script")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/configs_embeddings.yaml",
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--embeddings_root_dir",
        type=str,
        required=True,
        help="Path to the embeddings directory",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force the use of CPU even if GPU is available",
    )
    return parser.parse_args()


def generate_embeddings(
    data_loader: DataLoader, model: nn.Module, embeddings_root_dir: Path
):
    embeddings_root_dir = Path(embeddings_root_dir)

    for x, y, paths in tqdm(data_loader, total=len(data_loader)):
        x = x.to(device)
        with torch.no_grad():
            embeddings = model(x)

        for embedding, label, path in zip(embeddings, y, paths):
            cls_dir = embeddings_root_dir / str(label.item())
            cls_dir.mkdir(parents=True, exist_ok=True)
            np.save(cls_dir / f"{path.stem}.npy", embedding.cpu().numpy())


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    model = models.get_model(
        model_name=config["model"]["model_name"],
        checkpoint_path=config["model"]["checkpoint_path"],
    )
    model.to(device)
    model.eval()

    embeddings_root_dir = Path(args.embeddings_root_dir)
    embeddings_root_dir.mkdir(parents=True, exist_ok=True)

    # Data
    path_mlaad = config["data"]["mlaad_root_path"]
    path_protocols = Path(config["data"]["protocols_root_path"])

    protocols = {
        "train": path_protocols / "train.csv",
        "dev": path_protocols / "dev.csv",
        "test": path_protocols / "eval.csv",
    }

    for subset, protocols_root_path in protocols.items():
        assert protocols_root_path.exists(), f"{protocols_root_path} does not exist"

    dataframes = {
        subset: pd.read_csv(protocols_root_path)
        for subset, protocols_root_path in protocols.items()
    }

    # Concat all datasets to transform model names into model ids and as new column do each df
    all_df = pd.concat(dataframes.values())
    le = LabelEncoder()
    le.fit(all_df["model_name"])
    class_mapping = {name: idx for idx, name in enumerate(le.classes_)}

    for subset, df in dataframes.items():
        df["model_id"] = le.transform(df["model_name"])

    for subset, df in dataframes.items():
        print(f"Generating '{subset}' subset embeddings")
        dataset = BaseDataset(
            basepath=path_mlaad,
            sr=config["data"]["sampling_rate"],
            sample_length_s=config["data"]["sample_length_s"],
            meta_data=df.to_dict(orient="records"),
            class_mapping=class_mapping,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=config["data"]["batch_size"],
            collate_fn=dataset.collate_fn,
            shuffle=True,
            num_workers=config["data"]["num_workers"],
        )
        generate_embeddings(
            data_loader=data_loader,
            model=model,
            embeddings_root_dir=embeddings_root_dir / subset,
        )
