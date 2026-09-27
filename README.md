# Plant Disease Detection

Deep Learning midterm project — phân loại bệnh trên lá cây (15 class, Pepper / Potato / Tomato) từ bộ dữ liệu [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease), so sánh 3 model có độ phức tạp tăng dần:

| Model | Kiến trúc | Notebook |
|---|---|---|
| Model 1 | Simple Sequential CNN (baseline) | `notebooks/02_model1_simple_cnn.ipynb` |
| Model 2 | Inception-style / deeper CNN | `notebooks/03_model2_inception.ipynb` |
| Model 3 | Transfer learning (MobileNetV2 / ResNet, ImageNet) | `notebooks/04_model3_transfer.ipynb` |

## Cấu trúc thư mục

```
Plant_disease/
├── data/                  # (gitignored) dữ liệu ảnh
│   ├── raw/PlantVillage/  #   ảnh gốc, mỗi thư mục = 1 class
│   └── processed/         #   dữ liệu trung gian (nếu cần)
├── splits/                # train/val/test split cố định (commit lên git, cả nhóm dùng chung)
├── notebooks/             # 01_eda, 02–04 cho từng model, 05_comparison
├── src/                   # code dùng chung: load dữ liệu, tiền xử lý, đánh giá
│   └── models/            # định nghĩa kiến trúc từng model
├── models/                # (gitignored) weights đã train (.keras/.h5)
├── outputs/
│   ├── figures/           # biểu đồ: training curves, confusion matrix, ảnh mẫu
│   ├── metrics/           # kết quả đánh giá (json/csv) của từng model
│   └── logs/              # (gitignored) TensorBoard / training logs
├── reports/               # báo cáo
├── slides/                # slide bảo vệ
└── requirements.txt
```

## Quy ước chung

- **Dùng chung một split**: mọi model phải load dữ liệu qua `splits/` — không tự chia lại, để kết quả so sánh công bằng.
- **Seed cố định**: `42`.
- **Đánh giá** trên test set bằng hàm chung trong `src/`, lưu kết quả vào `outputs/metrics/<model_name>.json` và hình vào `outputs/figures/`.
- Weights đã train không commit lên git — chia sẻ qua Google Drive.

## Cài đặt

Python **3.10** (TensorFlow 2.10 không hỗ trợ 3.11+).

```bash
python -m venv .venv-tf
.venv-tf\Scripts\activate          # Windows
pip install -r requirements.txt
```

Đặt dataset vào `data/raw/PlantVillage/` (tải từ Kaggle, chỉ giữ 15 class Pepper/Potato/Tomato).

> Trên Windows native, TensorFlow ≥ 2.11 không hỗ trợ GPU — vì vậy dự án dùng TF 2.10.1 kèm CUDA/cuDNN cài qua pip. Trên Colab/Kaggle chỉ cần `pip install tensorflow==2.10.1`.
