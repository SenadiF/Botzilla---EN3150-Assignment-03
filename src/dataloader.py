from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from .dataset import RealWasteDataset


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
manifest = DATA_DIR / "realwaste_data.csv"
dataset_root = DATA_DIR / "RealWasteData/realwaste-main/RealWaste"

base = RealWasteDataset(manifest, dataset_root)
# targets
y = base.targets


# transfrom to 64*64
train_tf = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# transfrom without augemntation to 64*64
eval_tf = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# train test split 
idx_train, idx_temp = train_test_split(
    range(len(y)), test_size = 0.3, stratify=y, random_state=42
)

y_temp = [y[i] for i in idx_temp]

idx_val, idx_test = train_test_split(
    idx_temp, test_size=0.50, stratify=y_temp, random_state=42
)


# Dataset - train with augmentation, val/test without 
train_ds = Subset(RealWasteDataset(manifest, dataset_root, transform=train_tf), idx_train)
val_ds = Subset(RealWasteDataset(manifest, dataset_root, transform=eval_tf), idx_val)
test_ds = Subset(RealWasteDataset(manifest, dataset_root, transform=eval_tf), idx_test)
print(f"Traing size : {len(train_ds)}")
print(f"validation size : {len(val_ds)}")
print(f"tast size {len(test_ds)}")

# dataloder 
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

