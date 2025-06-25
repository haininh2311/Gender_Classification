import unittest
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from src.inference.predictor import Predictor
from src.data.loader import GenderDataset
from src.data.transform import get_transforms


class TestPredictorFast(unittest.TestCase):
    def setUp(self):
        self.model_path = "experiments/checkpoints/best_model.pth"
        self.data_root = "datasets"
        self.transforms = get_transforms()
        self.batch_size = 32

        # Tạo dataset và dataloader thủ công cho train và val
        train_dataset = GenderDataset(
            Path(self.data_root) / "Train", transform=self.transforms["train"]
        )
        val_dataset = GenderDataset(
            Path(self.data_root) / "Val", transform=self.transforms["val"]
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        # Load model
        self.predictor = Predictor(model_path=self.model_path)
        self.model = self.predictor.model  # Đã load weight, eval mode
        print("Mô hình đã load thành công!")

    def _evaluate_loader(self, loader):
        y_true, y_pred = [], []
        print("Bắt đầu đánh giá...")
        for images, labels in loader:
            with torch.no_grad():
                outputs = self.model(images)
                preds = outputs.argmax(dim=1)
                y_true.extend(labels.tolist())
                y_pred.extend(preds.tolist())
        print("Đánh giá hoàn tất!")
        return accuracy_score(y_true, y_pred)

    def test_predictor_on_train_set(self):
        print("\nĐánh giá trên tập TRAIN:")
        acc = self._evaluate_loader(self.train_loader)
        print(f"Train accuracy: {acc * 100:.2f}%")
        self.assertGreater(acc, 0.9)

    def test_predictor_on_val_set(self):
        print("\nĐánh giá trên tập VAL:")
        acc = self._evaluate_loader(self.val_loader)
        print(f"Val accuracy: {acc * 100:.2f}%")
        self.assertGreater(acc, 0.9)


if __name__ == "__main__":
    unittest.main()
