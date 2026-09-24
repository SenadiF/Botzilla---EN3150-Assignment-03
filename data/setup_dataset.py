import requests
import zipfile
from pathlib import Path
from tqdm import tqdm


# Direct UCI download URL
URL = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"

# Where to save the dataset
DATASET_ROOT = Path("RealWasteData")
ZIP_PATH = DATASET_ROOT / "realwaste.zip"
EXTRACT_PATH = DATASET_ROOT


def download_file(url: str, destination: Path):
    """Download a file with a progress bar."""

    destination.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset...")
    print(f"URL: {url}")
    print(f"Saving to: {destination}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024 * 1024  # 1 MB

    with open(destination, "wb") as file:
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:

            for data in response.iter_content(block_size):
                file.write(data)
                progress.update(len(data))

    print("\nDownload completed!")


def extract_dataset(zip_path: Path, extract_path: Path):
    """Extract the ZIP dataset."""

    print("\nExtracting dataset...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    print("Extraction completed!")


def main():

    # Skip downloading if ZIP already exists
    if not ZIP_PATH.exists():
        download_file(URL, ZIP_PATH)
    else:
        print(f"Dataset ZIP already exists: {ZIP_PATH}")

    # Extract dataset
    extract_dataset(ZIP_PATH, EXTRACT_PATH)

    print("\nDataset ready!")
    print(f"Location: {EXTRACT_PATH.resolve()}")


if __name__ == "__main__":
    main()