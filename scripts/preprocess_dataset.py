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
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.dataset import MLAADBaseDataset
from src.datasets.utils import HuggingFaceFeatureExtractor, WaveformEmphasiser


def parse_args():
    parser = argparse.ArgumentParser(description="Data augmentation script")
    # Datasets and protocols
    parser.add_argument(
        "--mlaad_path",
        type=str,
        default="data/MLAAD_v5/",
        help="Path to MLAADv5 dataset",
    )
    parser.add_argument(
        "--protocol_path",
        type=str,
        default="data/MLAADv5_for_sourcetracing/",
        help="Path to MLAADv5 protocols",
    )
    parser.add_argument(
        "--musan_path",
        type=str,
        default="data/musan/",
        help="Path to the MUSAN dataset",
    )
    parser.add_argument(
        "--rir_path",
        type=str,
        default="data/RIRS_NOISES/",
        help="Path to RIRs dataset",
    )

    # HuggingFace feature extractor
    parser.add_argument(
        "--model_name",
        type=str,
        default="wav2vec2-base",
        help="name of the feature extractor",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="Wav2Vec2Model",
        help="Class of the feature extractor",
    )
    parser.add_argument(
        "--model_layer",
        type=int,
        default=5,
        help="Which layer to use from the feature extractor",
    )
    parser.add_argument(
        "--hugging_face_path",
        type=str,
        default="facebook/wav2vec2-base",
        help="Path from the HF collections",
    )
    parser.add_argument(
        "--sampling_rate", type=int, default=16_000, help="Audio sampling rate"
    )
    parser.add_argument(
        "--max_length", type=int, default=4, help="Crop the audio to X seconds"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for preprocessing"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Workers for loaders"
    )

    # Output folder
    parser.add_argument(
        "--out_folder", type=str, default="exp", help="Where to write the results"
    )
    args = parser.parse_args()
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
    return args


def main(args):

    # Read the MLAAD data
    path_mlaad = args.mlaad_path
    path_protocols = args.protocol_path
    train_protocol = os.path.join(path_protocols, "train.csv")
    dev_protocol = os.path.join(path_protocols, "dev.csv")
    test_protocol = os.path.join(path_protocols, "eval.csv")
    assert os.path.exists(train_protocol), f"{train_protocol} does not exist"
    assert os.path.exists(dev_protocol), f"{dev_protocol} does not exist"
    assert os.path.exists(test_protocol), f"{test_protocol} does not exist"
    train_df = pd.read_csv(train_protocol)
    dev_df = pd.read_csv(dev_protocol)
    test_df = pd.read_csv(test_protocol)

    # Encode the system names to unique int values
    # Use only the training data classes. The others are OOD
    le = LabelEncoder()
    le.fit(train_df["model_name"])
    train_df["model_id"] = le.transform(train_df["model_name"])
    class_mapping = {name: [idx, "ID"] for idx, name in enumerate(le.classes_)}

    # Add a OOD label for unseen systems in the training data
    for k in pd.concat([dev_df["model_name"], test_df["model_name"]]):
        if k not in class_mapping:
            class_mapping[k] = [len(class_mapping), "OOD"]

    # Save the label assignment
    with open(os.path.join(args.out_folder, "label_assignment.txt"), "w") as fout:
        for k, v in sorted(
            class_mapping.items(), key=lambda item: (item[1], item[0].lower())
        ):
            fout.write(f"{k.ljust(50)}|{str(v[0]).ljust(3)}|{v[1]}\n")
        print(
            f"[INFO] Label assignment written to: {args.out_folder}/label_assignment.txt"
        )

    # Prepare dataloaders
    train_data = MLAADBaseDataset(
        basepath=path_mlaad,
        sr=args.sampling_rate,
        sample_length_s=args.max_length,
        meta_data=train_df.to_dict(orient="records"),
        class_mapping=class_mapping,
        max_samples=-1,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        collate_fn=train_data.collate_fn,
        shuffle=False,
        num_workers=args.num_workers,
    )

    dev_data = MLAADBaseDataset(
        basepath=path_mlaad,
        sr=args.sampling_rate,
        sample_length_s=args.max_length,
        meta_data=dev_df.to_dict(orient="records"),
        class_mapping=class_mapping,
        max_samples=-1,
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=args.batch_size,
        collate_fn=train_data.collate_fn,
        shuffle=False,
        num_workers=args.num_workers,
    )

    test_data = MLAADBaseDataset(
        basepath=path_mlaad,
        sr=args.sampling_rate,
        sample_length_s=args.max_length,
        meta_data=test_df.to_dict(orient="records"),
        class_mapping=class_mapping,
        max_samples=-1,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        collate_fn=train_data.collate_fn,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Load the feature extractor
    feature_extractor = HuggingFaceFeatureExtractor(
        model_class_name=args.model_class,
        layer=args.model_layer,
        name=args.hugging_face_path,
    )

    ## Run the augmentation
    list_of_emphases = ["original", "reverb", "speech", "music", "noise"]
    emphasiser = WaveformEmphasiser(args.sampling_rate, args.musan_path, args.rir_path)
    for subset_, loader in zip(
        ["train", "dev", "eval"], [train_loader, dev_loader, test_loader]
    ):
        count = 0
        feature_folder = os.path.join(args.out_folder, "preprocess_" + args.model_name)
        target_dir = os.path.join(feature_folder, subset_)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        print(f"[INFO] Processing {subset_} data...")
        print(f"[INFO] Writing features to {target_dir}")
        for waveform, label, file_name in tqdm(loader):
            for emphasis in list_of_emphases:
                waveform = emphasiser(waveform, emphasis)
                hidden_state = feature_extractor(waveform, args.sampling_rate)

                # Create a unique filename which also includes the class id
                # i.e. 000001_class_emphasisType_originalFileName.pt
                orig_file_name = os.path.splitext(os.path.split(file_name[0])[1])[0]
                out_file_name = (
                    f"{count:06d}_{label.item()}_{emphasis}_{orig_file_name}.pt"
                )
                torch.save(
                    hidden_state.float(), os.path.join(target_dir, out_file_name)
                )
                count += 1
    print("[INFO] Augmentation step finished")


if __name__ == "__main__":
    args = parse_args()
    main(args)
