import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def show_images(batch, title: str | None = None):
    """Display a batch of images (tensor) in a grid."""
    grid_img = make_grid(batch, nrow=8).permute(1, 2, 0).numpy()
    plt.figure(figsize=(12, 6))
    if title:
        plt.title(title)
    plt.imshow(grid_img)
    plt.axis("off")
    plt.show()


def show_prediction(img, label: str, pred: str):
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(f"GT: {label} | Pred: {pred}")
    plt.axis("off")
    plt.show()
