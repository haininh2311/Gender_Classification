from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import os


class GenderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for label, gender in enumerate(["Male", "Female"]):
            folder = Path(root_dir) / gender
            for img in os.listdir(folder):
                if img.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((folder / img, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloaders(
    data_root, transforms_dict, batch_size=32, num_workers=4, phases=None
):
    if phases is None:
        phases = ["train", "val", "test"]
    loaders = {}
    for phase in phases:
        dataset = GenderDataset(
            Path(data_root) / phase, transform=transforms_dict[phase]
        )
        loaders[phase] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(phase == "train"),
            num_workers=num_workers,
        )
    return loaders
