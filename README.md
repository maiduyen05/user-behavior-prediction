# 🏆 User Behavior Prediction — DATAFLOW 2026

> **Bán kết DATAFLOW 2026 · The Alchemy of Minds**  
> Câu lạc bộ Toán Tin HAMIC — Trường Đại học Khoa học Tự nhiên, ĐHQGHN

---

## 📋 Tổng quan bài toán

Xây dựng mô hình **Multi-output Classifier** dự đoán **6 thuộc tính hành vi độc lập** của khách hàng trong tương lai, dựa trên chuỗi hành động tương tác trong quá khứ (được mã hóa thành số định danh).

| Hạng mục | Chi tiết |
|---|---|
| **Task** | Multi-output Classification |
| **Input** | Chuỗi hành động mã hóa, độ dài biến thiên (Variable-length sequence) |
| **Output** | 6 thuộc tính hành vi (attr\_1 → attr\_6), định dạng UINT16 |
| **Metric** | Exact-Match Accuracy (đúng đồng thời cả 6 thuộc tính) |

---

## 📁 Cấu trúc dự án

```
.
├── data/
│   ├── X_train.csv                  # Chuỗi hành động — tập huấn luyện
│   ├── Y_train.csv                  # Nhãn 6 thuộc tính — tập huấn luyện
│   ├── X_val.csv                    # Chuỗi hành động — tập kiểm định
│   ├── Y_val.csv                    # Nhãn 6 thuộc tính — tập kiểm định
│   └── X_test.csv                   # Chuỗi hành động — tập kiểm thử (không có nhãn)
│
├── src/
│   ├── 01_eda.ipynb                 # Phân tích khám phá dữ liệu (EDA)
│   ├── 02_lightGBM_model.ipynb      # Mô hình LightGBM (baseline)
│   ├── 03_gru_model.ipynb           # Mô hình GRU
│   └── 04_transformer_model.ipynb   # Mô hình Transformer (best)
│
├── submissions/
│   ├── submission_lgbm.csv          # Kết quả dự đoán — LightGBM
│   ├── submission_gru.csv           # Kết quả dự đoán — GRU
│   └── submission_transformer.csv   # Kết quả dự đoán — Transformer ⭐
│
├── requirements.txt
└── README.md
```

---

## 🚀 Hướng dẫn cài đặt

### Yêu cầu môi trường

```
Python >= 3.9
CUDA >= 11.8 (nếu chạy trên GPU)
```

### Cài đặt thư viện

```bash
pip install -r requirements.txt
```

> 🖥️ **Nếu dùng GPU (CUDA 11.8)**, cài PyTorch riêng trước để chọn đúng phiên bản CUDA, sau đó mới chạy requirements:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> pip install -r requirements.txt
> ```

---

## ▶️ Tái tạo kết quả

### Bước 1 — Phân tích dữ liệu

Notebook EDA được thiết lập **đường dẫn tương đối**, chạy trực tiếp trên máy local mà không cần chỉnh sửa:

```bash
src/01_eda.ipynb
```

### Bước 2 — Huấn luyện mô hình

> ⚠️ **Lưu ý quan trọng**: Các notebook huấn luyện (`02`, `03`, `04`) yêu cầu GPU và **không thể chạy trên máy local thông thường**. Nhóm thực hiện huấn luyện trên **Kaggle Notebook**. Khi BTC kiểm tra, vui lòng:
> 1. Import notebook lên Kaggle
> 2. Upload dữ liệu lên Kaggle Dataset
> 3. Cập nhật đường dẫn trong phần `CFG` ở đầu mỗi notebook thành đường dẫn dataset tương ứng trên Kaggle

Mỗi notebook thực hiện đầy đủ 2 giai đoạn:
- **Giai đoạn 1**: Train trên `X_train/Y_train`, đánh giá trên `X_val/Y_val`
- **Giai đoạn 2**: Gộp Train + Val, retrain toàn bộ, predict `X_test` → xuất file submission


### Bước 3 — Kết quả dự đoán

File submission cuối nằm tại `submissions/submission_transformer.csv` với định dạng:

```
id, attr_1, attr_2, attr_3, attr_4, attr_5, attr_6
```

---

## 🧠 Mô hình và kết quả

| Mô hình | Val Exact-Match Accuracy | Ghi chú |
|---|---|---|
| LightGBM | — | Baseline, feature engineering thủ công |
| GRU | — | Sequence model, xử lý chuỗi tuần tự |
| **Transformer** | **—** | ⭐ Mô hình cuối, kiến trúc tốt nhất |


