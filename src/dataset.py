# custom data set
import os
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

class RealWasteDataset(Dataset):
    """
    custom datset.
        data_file - data/realwaste_data.csv
        image_path -  data/RealWasteData/realwaste-main/RealWaste
    """
    def __init__(self, data_file, image_path, transform=None):
        self.image_path = Path(image_path)
        self.image_lables = pd.read_csv(data_file)
        self.transform = transform

        # label -> class name mapping, taken from the manifest
        pairs = self.image_lables[["label", "class_name"]].drop_duplicates().sort_values("label")
        self.classes = pairs.class_name.tolist()
        self.class_to_idx = dict(zip(pairs.class_name, pairs.label))

    def __len__(self):
        return len(self.image_lables)

    def __getitem__(self, index):
        row = self.image_lables.iloc[index]
        image = Image.open(self.image_path / row.filepath).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row.label)

    @property
    def targets(self):
        """All labels in manifest order, for stratifying the split."""
        return self.image_lables.label.tolist()
