"""
DISCLAIMER:
This code is provided "as-is" without any warranty of any kind, either expressed or implied,
including but not limited to the implied warranties of merchantability and fitness for a particular purpose.
The author assumes no liability for any damages or consequences resulting from the use of this code.
Use it at your own risk.

Utility to download and extract all resources needed for the MLAADv5 project.

This script handles the downloading of large files with progress bars, ensures
caching of already downloaded files, and extracts `.zip` files using 7-Zip.

## Author: Nicolas MUELLER
## December 2024
"""

import sys
from pathlib import Path

# Enables running the script from root directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import subprocess

import requests
from tqdm import tqdm


def download_file(session, file_url, save_path):
    """
    Download a file with a progress bar.

    Parameters:
        session (requests.Session): The HTTP session for downloading.
        file_url (str): The URL of the file to download.
        save_path (str): The local path where the file should be saved.

    Returns:
        None
    """
    # Check if the file exists
    if os.path.exists(save_path):
        print(f"File already exists: {save_path}")
        return

    # Get the file size from headers
    response = session.head(file_url, allow_redirects=True)
    if response.status_code != 200:
        print(
            f"Failed to fetch headers for: {file_url}, status code: {response.status_code}"
        )
        return

    file_size = int(response.headers.get("content-length", 0))

    # Start downloading the file with a progress bar
    response = session.get(file_url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as file, tqdm(
            desc=f"Downloading {os.path.basename(save_path)}",
            total=file_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                progress_bar.update(len(chunk))
        print(f"Download completed: {save_path}")
    else:
        print(f"Failed to download: {file_url}, status code: {response.status_code}")


def extract_zip_file(zip_path, extract_dir):
    """
    Extract a `.zip` file using 7-Zip.

    Parameters:
        zip_path (str): The path to the `.zip` file to be extracted.
        extract_dir (str): The directory where the files will be extracted.

    Returns:
        None
    """
    print(f"Extracting {zip_path} to {extract_dir}...")
    try:
        subprocess.run(["7za", "x", zip_path, f"-o{extract_dir}"], check=True)
        print(f"Extraction completed: {zip_path}")
    except subprocess.CalledProcessError as e:
        print(f"Extraction failed: {e}")
    except FileNotFoundError:
        print("7-Zip (7za) is not installed. Please install it to enable extraction.")


def download_MLAADv5(save_dir):
    """
    Download and extract MLAADv5.
    Files are saved in the specified directory, and existing files are skipped.
    """
    # MLAADv5 dataset file URLs
    files_mlaad = [
        f"https://owncloud.fraunhofer.de/index.php/s/tL2Y1FKrWiX4ZtP/download?path=%2Fv5&files=mlaad_v5.z0{i}"
        for i in range(1, 10)
    ] + [
        f"https://owncloud.fraunhofer.de/index.php/s/tL2Y1FKrWiX4ZtP/download?path=%2Fv5&files=mlaad_v5.z10",
        "https://owncloud.fraunhofer.de/index.php/s/tL2Y1FKrWiX4ZtP/download?path=%2Fv5&files=mlaad_v5.zip",
        "https://owncloud.fraunhofer.de/index.php/s/tL2Y1FKrWiX4ZtP/download?path=%2Fv5&files=mlaad_v5.zip.md5",
    ]

    # Directory to save downloaded files
    os.makedirs(save_dir, exist_ok=True)

    # Create a session
    session = requests.Session()

    # Download dataset and protocol files
    all_files = files_mlaad
    for file_url in all_files:
        # Extract file name from URL
        file_name = os.path.basename(file_url.split("&files=")[-1])
        save_path = os.path.join(save_dir, file_name)
        # Download file if not cached
        download_file(session, file_url, save_path)

    print(f"Now, go into the directory {save_dir} and perform:")
    print(f"md5sum -c mlaad_v5.zip.md5")
    print(f"7za x mlaad_v5.zip -o./")


if __name__ == "__main__":
    # Download MLAADv5 dataset
    download_MLAADv5("data/MLAADv5")

    # Protocol file URL
    save_protocols_path = "data/MLAADv5_for_sourcetracing/mlaadv5_for_sourcetracing.zip"
    os.makedirs(os.path.dirname(save_protocols_path), exist_ok=True)
    if not os.path.exists(save_protocols_path):
        url_protocols = "https://deepfake-total.com/data/mlaad4sourcetracing.zip"
        download_file(requests.Session(), url_protocols, save_protocols_path)
