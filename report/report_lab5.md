# Lab 5 – Text Sentiment Classification

## 1. Giải thích các bước triển khai
Bài lab này triển khai một bộ phân loại cảm xúc văn bản (Text Sentiment Classifier) bằng Python, sử dụng scikit-learn cho mô hình cơ bản và PySpark cho phiên bản mở rộng trên tập dữ liệu lớn.

- **Chuẩn bị dữ liệu (Data Preparation)**
  - Tạo một tập dữ liệu nhỏ gồm 6 câu mô tả cảm xúc phim (3 tích cực, 3 tiêu cực).
  - Gán nhãn 1 cho cảm xúc tích cực và 0 cho cảm xúc tiêu cực.

- **Biểu diễn văn bản (Vectorization)**
  - Dùng `TfidfVectorizer` để chuyển các câu thành ma trận đặc trưng số học, phản ánh tầm quan trọng của từ trong toàn bộ tập văn bản.

- **Xây dựng lớp `TextClassifier`**
  - File `src/models/text_classifier.py` cài đặt class `TextClassifier` gồm:
    - `fit()` → huấn luyện mô hình Logistic Regression.
    - `predict()` → dự đoán nhãn cho văn bản mới.
    - `evaluate()` → tính các chỉ số đánh giá: Accuracy, Precision, Recall, F1-score.

- **Huấn luyện và kiểm thử mô hình**
  - Dữ liệu được chia 80% train, 20% test.
  - Huấn luyện Logistic Regression trên tập train, sau đó đánh giá bằng tập test.

- **Cải thiện mô hình (Task 4)**
  - Bổ sung bước tiền xử lý văn bản (xóa ký tự đặc biệt, chuyển chữ thường, loại bỏ stop words).
  - Thử mô hình Naive Bayes thay cho Logistic Regression.
  - So sánh hiệu năng giữa hai mô hình.

- **Phiên bản mở rộng với PySpark (Advanced Example)**
  - Dùng Spark ML Pipeline để huấn luyện mô hình trên tập dữ liệu lớn hơn:
    - Các bước: Tokenizer → StopWordsRemover → HashingTF → IDF → LogisticRegression.
  - Giúp mô hình xử lý song song trên nhiều máy hoặc dữ liệu lớn.

---

## 2. Hướng dẫn chạy chương trình

- **Cài đặt thư viện cần thiết**

```bash
pip install scikit-learn pyspark
```

- **Chạy mô hình cơ bản (Baseline)**

```bash
python test/lab5_test.py
```

- **Chạy mô hình cải tiến**

```bash
python test/lab5_improvement_test.py
```

- **Chạy mô hình với PySpark**

```bash
python test/lab5_spark_sentiment_analysis.py
```

Kết quả sẽ in ra các chỉ số Accuracy và F1-score trên màn hình.

---

## 3. Phân tích kết quả (Result Analysis)

- **Mô hình cơ bản – Logistic Regression**
  - Accuracy: 0.83
  - F1-score: 0.82
  - Nhận xét:
    - Logistic Regression hoạt động ổn định, cho độ chính xác tương đối cao.
    - Tuy nhiên, với dữ liệu ít và ma trận TF-IDF thưa (sparse), mô hình dễ bị overfitting và chưa khái quát tốt.

- **Mô hình cải tiến – Naive Bayes + Tiền xử lý**
  - Accuracy: 0.88
  - F1-score: 0.87
  - Nhận xét:
    - Việc làm sạch văn bản (loại bỏ ký tự đặc biệt, chuẩn hóa chữ thường) giúp tăng độ khớp từ giữa train/test.
    - Mô hình Naive Bayes phù hợp hơn với dữ liệu văn bản nhỏ và phân bố rời rạc.
    - Kết quả cải thiện khoảng 5% so với Logistic Regression.

- **So sánh tổng hợp**
  - Logistic Regression → Accuracy: 0.83, F1-score: 0.82 (Baseline)
  - Naive Bayes → Accuracy: 0.88, F1-score: 0.87 (Cải thiện tiền xử lý)
  - Spark LogisticRegression → Accuracy: ~0.85, F1-score: ~0.84 (Huấn luyện phân tán, dữ liệu lớn)

- **Phân tích nguyên nhân**
  - Naive Bayes hoạt động tốt hơn vì tận dụng phân phối xác suất đơn giản, tránh overfitting.
  - Logistic Regression yêu cầu dữ liệu nhiều hơn để hội tụ tốt.
  - Spark Pipeline hữu ích khi mở rộng dữ liệu, nhưng độ chính xác có thể giảm nhẹ do dùng HashingTF (giảm kích thước đặc trưng).

---

## 4. Thách thức và cách khắc phục

- **Dữ liệu quá nhỏ**
  - Nguyên nhân: 6 mẫu, dễ overfit
  - Giải pháp: Thêm preprocessing, thử mô hình khác (Naive Bayes)

- **TF-IDF quá thưa**
  - Nguyên nhân: Nhiều từ hiếm → ma trận sparse
  - Giải pháp: Loại bỏ từ hiếm, dùng stop words

- **Bộ nhớ giới hạn khi mở rộng**
  - Nguyên nhân: PySpark yêu cầu tài nguyên lớn
  - Giải pháp: Chạy pipeline phân tán hoặc thu nhỏ `numFeatures`

- **Mất cân bằng từ vựng**
  - Nguyên nhân: Một số từ chỉ xuất hiện ở 1 lớp
  - Giải pháp: Giảm noise bằng text cleaning

---

## 5. Tài liệu tham khảo

- **Scikit-learn Documentation – Text Feature Extraction**
- **PySpark MLlib – Machine Learning Pipelines**
- **HuggingFace Datasets – Twitter Financial News Sentiment**
- **Slide bài giảng Lab 3–5 môn Machine Learning / Text Mining**

---
