# ----------------------------
# Gender Classification with ConvNeXt and Flask

## Mô tả
Phân loại giới tính từ ảnh khuôn mặt bằng mô hình ConvNeXt và giao diện Flask đơn giản.

## Cài đặt
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Huấn luyện
Huấn luyện model với notebook: `notebooks/gender_classification.ipynb`. Kết quả được lưu tại:
```
experiments/checkpoints/best_model.pth
```

## Chạy Flask App
```bash
python -m src.gui.web_app
```
Truy cập tại [http://127.0.0.1:5000](http://127.0.0.1:5000)


