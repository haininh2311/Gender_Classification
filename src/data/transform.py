from torchvision import transforms


def get_transforms(img_size: int = 224):
    normalize_mean = [0.5, 0.5, 0.5]
    normalize_std = [0.5, 0.5, 0.5]

    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ]
    )

    val_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ]
    )

    return {"train": train_tf, "val": val_tf, "test": val_tf}
