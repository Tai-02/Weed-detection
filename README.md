# 🌿 Weed Detection — Phát Hiện Cỏ Dại Trong Nông Nghiệp

> Đồ án môn **Xử Lý Ảnh Số** — Ứng dụng CNN và SVM để phân loại và phát hiện cỏ dại trên ảnh nông nghiệp.

---

## 📌 Giới Thiệu

Cỏ dại cạnh tranh dinh dưỡng, nước và ánh sáng với cây trồng, gây giảm năng suất đáng kể. Dự án này xây dựng một hệ thống tự động nhận diện cỏ dại từ ảnh chụp thực địa bằng kỹ thuật xử lý ảnh kết hợp học sâu.

Hệ thống gồm hai bước chính:
1. **Phân loại ảnh** — dùng CNN hoặc SVM để phân biệt 4 lớp đối tượng.
2. **Phát hiện & định vị** — dùng phân đoạn Watershed để khoanh vùng từng cây và phân loại từng vùng.

---

## 🗂️ Dataset

| Lớp | Số ảnh gốc | Ghi chú |
|---|---|---|
| `broadleaf` | ~1.000 | Cỏ lá rộng — được tăng cường dữ liệu |
| `grass` | ~3.000 | Cỏ hòa thảo |
| `soil` | ~3.000 | Đất trống |
| `soybean` | ~3.000 | Cây đậu nành |

- Ảnh đầu vào: `128×128` pixel, RGB  
- Tỉ lệ chia: **Train 60% / Val 20% / Test 20%**  
- Lớp `broadleaf` được tăng cường bằng lật ngang/dọc để cân bằng dữ liệu

---

## ⚙️ Tiền Xử Lý Ảnh

Mỗi ảnh đầu vào được xử lý qua pipeline sau:

1. **Resize** về `128×128`
2. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) trong không gian màu LAB — tăng độ tương phản cục bộ
3. **Gaussian Blur** — giảm nhiễu
4. **Sharpening** — làm nét cạnh bằng kernel convolution
5. **Normalize** về `[0, 1]`

---

## 🧠 Mô Hình

### 1. CNN (Convolutional Neural Network)

```
Input (128×128×3)
→ Conv2D(32) + MaxPool
→ Conv2D(64) + MaxPool
→ Conv2D(128) + MaxPool
→ Flatten
→ Dense(128) + Dropout(0.3)
→ Dense(4, softmax)
```

**Callbacks:**
- `EarlyStopping` — dừng sớm khi `val_loss` không cải thiện sau 5 epoch
- `ReduceLROnPlateau` — giảm learning rate khi `val_loss` bão hòa
- `ModelCheckpoint` — lưu model tốt nhất theo `val_accuracy`

**Optimizer:** Adam | **Loss:** Sparse Categorical Crossentropy | **Epochs:** 35 | **Batch size:** 32

---

### 2. SVM (Support Vector Machine)

- Flatten ảnh → vector 1D
- Chuẩn hóa bằng `StandardScaler`
- Giảm chiều bằng **PCA** (300 thành phần)
- Huấn luyện **SVM** với kernel RBF (`C=10`, `gamma='scale'`)

---

## 🔍 Phát Hiện Vật Thể (Detection Pipeline)

Sau khi huấn luyện, hệ thống có thể định vị và phân loại từng cây trong ảnh thực địa:

```
Ảnh gốc
  │
  ├─ Tính chỉ số thực vật (ExG, ExGN, CIVE)
  │    └─ Tạo mask nhị phân vùng thực vật
  │
  ├─ Phân đoạn Watershed
  │    └─ Tách từng vùng thực vật riêng biệt
  │
  └─ Phân loại từng vùng bằng CNN
       └─ Vẽ bounding box (chỉ hiển thị grass & broadleaf)
```

**Chỉ số thực vật sử dụng:**
- **ExG** (Excess Green Index)
- **ExGN** (Normalized Excess Green)
- **CIVE** (Color Index of Vegetation Extraction)

---

## 📊 Đánh Giá

Kết quả được đánh giá qua:
- **Accuracy** trên tập Train / Val / Test
- **Classification Report** (Precision, Recall, F1-score từng lớp)
- **Confusion Matrix** trực quan hóa bằng matplotlib

---

## 🚀 Hướng Dẫn Chạy

### Yêu cầu

- Python 3.8+
- Google Colab (khuyến nghị) hoặc môi trường local có GPU

### Cài đặt thư viện

```bash
pip install tensorflow keras scikit-learn opencv-python pillow matplotlib scikit-image
```

### Chạy trên Google Colab

1. Upload notebook `DoAN_HM_XLAS.ipynb` lên Google Colab
2. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Đặt dataset vào đường dẫn:
   ```
   /content/drive/MyDrive/dataset/
   ├── broadleaf/
   ├── grass/
   ├── soil/
   └── soybean/
   ```
4. Chạy tuần tự các cell trong notebook

### Chạy Detection trên ảnh mới

Ở cuối notebook có cell cho phép upload ảnh từ máy tính và chạy toàn bộ pipeline phát hiện cỏ dại tự động.

---

## 📁 Cấu Trúc Dự Án

```
Weed-detection/
├── DoAN_HM_XLAS.ipynb      # Notebook chính (train + evaluate + detect)
├── super_cnn_model.keras   # Model CNN đã huấn luyện (sinh ra sau khi train)
└── README.md
```

---

## 🛠️ Công Nghệ Sử Dụng

| Thư viện | Mục đích |
|---|---|
| TensorFlow / Keras | Xây dựng và huấn luyện CNN |
| scikit-learn | SVM, PCA, đánh giá mô hình |
| OpenCV | Xử lý ảnh, Watershed, morphology |
| scikit-image | Phân tích region (regionprops) |
| Pillow | Đọc/xử lý ảnh |
| Matplotlib | Trực quan hóa kết quả |

---

## 👨‍💻 Tác Giả

Dự án thực hiện trong khuôn khổ đồ án môn **Xử Lý Ảnh Số**.

---

## 📄 Giấy Phép

Dự án phục vụ mục đích học thuật.
