from flask import Flask, render_template, request
import os
from src.inference.predictor import Predictor
from werkzeug.utils import secure_filename
import glob

# Thư mục thực tế chứa static/ (đường tuyệt đối)
STATIC_FOLDER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "static")
)
TEMPLATE_FOLDER = os.path.join(os.path.dirname(__file__), "templates")

app = Flask(__name__, static_folder=STATIC_FOLDER, template_folder=TEMPLATE_FOLDER)

predictor = Predictor(model_path="experiments/checkpoints/best_model.pth")


def clear_static_folder():
    """Xóa toàn bộ ảnh cũ trong thư mục static"""
    for f in glob.glob(os.path.join(STATIC_FOLDER, "*")):
        try:
            os.remove(f)
        except Exception as e:
            print(f"Không thể xóa {f}: {e}")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            clear_static_folder()
            filename = secure_filename(file.filename)

            # Đường dẫn tuyệt đối để lưu ảnh
            save_path = os.path.join(STATIC_FOLDER, filename)
            os.makedirs(STATIC_FOLDER, exist_ok=True)  # Đảm bảo thư mục tồn tại

            file.save(save_path)  # Lưu ảnh

            # Gửi đường dẫn đầy đủ cho predictor
            result = predictor.predict(save_path)

            # Truyền chỉ tên file cho template
            return render_template("index.html", prediction=result, image_path=filename)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
