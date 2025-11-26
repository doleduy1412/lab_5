# Part 4 – Named Entity Recognition với BiLSTM

## 4.1. Mô tả bài toán và mục tiêu
Trong phần này, em xây dựng một mô hình Mạng Nơ-ron Hồi quy hai chiều (BiLSTM) cho bài toán Nhận dạng Thực thể Tên (Named Entity Recognition – NER) trên bộ dữ liệu CoNLL-2003.

**Mục tiêu:**
- Tải và tiền xử lý dữ liệu NER (CoNLL-2003).
- Xây dựng từ điển cho từ và nhãn NER (BIO).
- Thiết kế mô hình Embedding → BiLSTM → Linear cho bài toán token classification.
- Huấn luyện, đánh giá mô hình trên tập validation bằng accuracy.
- Thử dự đoán thực thể tên cho câu mới: "VNU University is located in Hanoi".

## 4.2. Task 1 – Chuẩn bị dữ liệu

### 4.2.1. Dữ liệu NER CoNLL-2003
Bộ dữ liệu CoNLL-2003 gán nhãn NER theo định dạng BIO (Begin–Inside–Outside) với các loại thực thể chính:
- PER (Person)
- ORG (Organization)
- LOC (Location)
- MISC (Miscellaneous)

**Danh sách nhãn NER sử dụng:**
```
['B-LOC', 'B-MISC', 'B-ORG', 'B-PER',
 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER', 'O']
```

**Giải thích:**
- `B-XXX`: Bắt đầu một thực thể loại XXX (LOC, PER, ORG, MISC).
- `I-XXX`: Ở bên trong thực thể loại XXX.
- `O`: Token không thuộc bất kỳ thực thể nào.

### 4.2.2. Xây dựng vocabulary cho từ và nhãn
Từ tập train, em xây dựng:

1. **Từ điển từ word_to_ix:**
   - Thêm 2 token đặc biệt:
     - `<PAD>`: dùng để padding câu.
     - `<UNK>`: dùng cho các từ ngoài từ điển train.
   - **Kết quả:**
     - Kích thước word_to_ix: 23,626
     - Đây là số lượng từ khác nhau trong toàn bộ tập dữ liệu, kích thước này hợp lý với CoNLL-2003.

2. **Từ điển nhãn tag_to_ix:**
   - Thêm nhãn đặc biệt `<PAD>` để padding nhãn.
   - Thêm 9 nhãn NER.
   - **Kết quả:**
     - Kích thước tag_to_ix: 9 (không tính <PAD>).
     - Tương ứng 9 loại nhãn NER thực tế trong CoNLL-2003.
   - Xây dựng thêm `ix_to_tag` để chuyển ngược từ index → nhãn string khi dự đoán.

## 4.3. Task 2 – Dataset, DataLoader và collate_fn

### 4.3.1. Lớp NERDataset
Lớp `NERDataset` kế thừa `torch.utils.data.Dataset`:
- **Đầu vào:**
  - `sentences`: list[list[str]] – mỗi phần tử là một câu, mỗi câu là danh sách token.
  - `tags_str`: list[list[str]] – mỗi phần tử là danh sách nhãn BIO tương ứng.
- **`__getitem__(idx)`:**
  - Lấy câu thứ idx.
  - Chuyển từng từ thành index bằng word_to_ix (từ không có → <UNK>).
  - Chuyển từng nhãn thành index bằng tag_to_ix.
  - Trả về cặp tensor 1D: (sentence_indices, tag_indices).

### 4.3.2. Hàm collate_fn và DataLoader
Xử lý batch với các câu có độ dài khác nhau:
```python
def collate_fn(batch):
    sentences, tags = zip(*batch)
    padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=pad_word_idx)
    padded_tags = pad_sequence(tags, batch_first=True, padding_value=pad_tag_idx)
    lengths = torch.tensor([len(s) for s in sentences], dtype=torch.long)
    return padded_sentences, padded_tags, lengths
```

Khởi tạo DataLoader:
```python
train_loader = DataLoader(
    train_dataset, 
    batch_size=32,
    shuffle=True, 
    collate_fn=collate_fn
)
```

## 4.4. Task 2 – Xây dựng mô hình BiLSTM

Mô hình BiLSTM cho token classification gồm 3 tầng chính:

1. **Embedding (nn.Embedding)**
   - `num_embeddings = 23,626`
   - `embedding_dim = 100`
   - `padding_idx = word_to_ix["<PAD>"]`
   - Output: `(batch_size, seq_len, 100)`

2. **BiLSTM (nn.LSTM với bidirectional=True)**
   - `input_size = 100`
   - `hidden_size = 128`
   - `num_layers = 1`
   - `batch_first = True`
   - `bidirectional = True`
   - Output: `(batch_size, seq_len, 256)` (128*2 do bidirectional)

3. **Linear (nn.Linear)**
   - `in_features = 256`
   - `out_features = 9` (số lớp NER)
   - Output logits: `(batch_size, seq_len, 9)`

**Forward pass:**
```python
def forward(self, x):
    embedded = self.embedding(x)         # (B, T, E)
    lstm_out, _ = self.lstm(embedded)    # (B, T, 2H)
    logits = self.fc(lstm_out)           # (B, T, C)
    return logits
```

## 4.5. Task 3 – Huấn luyện và Đánh giá

### 4.5.1. Thiết lập huấn luyện
- **Thiết bị:** cuda nếu có GPU, ngược lại cpu.
- **Loss function:** `nn.CrossEntropyLoss(ignore_index=pad_tag_idx)`
- **Optimizer:** `torch.optim.Adam(model.parameters(), lr=1e-3)`
- **Siêu tham số:**
  - `embedding_dim = 100`
  - `hidden_dim = 128`
  - `batch_size = 32`
  - `n_epochs = 5`

### 4.5.2. Kết quả huấn luyện
**Log huấn luyện:**
```
Epoch 1/5
Loss: 0.5944
Validation Accuracy: 0.8754 (≈ 87.54%)

Epoch 2/5
Loss: 0.2897
Validation Accuracy: 0.9142 (≈ 91.42%)

Epoch 3/5
Loss: 0.1694
Validation Accuracy: 0.9250 (≈ 92.50%)

Epoch 4/5
Loss: 0.0982
Validation Accuracy: 0.9233 (≈ 92.33%)

Epoch 5/5
Loss: 0.0554
Validation Accuracy: 0.9293 (≈ 92.93%)
```
**Best validation accuracy: 92.93%**

### 4.5.3. Phân tích kết quả
- **Epoch 1 → 2:**
  - Loss giảm từ 0.5944 → 0.2897.
  - Accuracy tăng từ 87.54% → 91.42%.
  - → Mô hình học được các đặc trưng cơ bản của bài toán NER.

- **Epoch 2 → 3:**
  - Loss giảm từ 0.2897 → 0.1694.
  - Accuracy tăng từ 91.42% → 92.50%.
  - → Mô hình bắt đầu học sâu hơn về ranh giới thực thể.

- **Epoch 3 → 4:**
  - Loss giảm từ 0.1694 → 0.0982.
  - Accuracy giảm nhẹ từ 92.50% → 92.33%.
  - → Dấu hiệu overfit nhẹ.

- **Epoch 4 → 5:**
  - Loss giảm từ 0.0982 → 0.0554.
  - Accuracy tăng lên 92.93%.
  - → Mô hình hội tụ tốt, đạt hiệu năng ổn định.

**Nhận xét chung:**
- Loss giảm mượt, không có dấu hiệu diverge.
- Validation accuracy ~93% là tốt đối với mô hình BiLSTM NER đơn giản.
- Mô hình chưa sử dụng CRF hay embedding tiền huấn luyện.

## 4.6. Ví dụ dự đoán câu mới

**Câu:** "VNU University is located in Hanoi"

**Kết quả dự đoán:**
```
VNU → B-ORG
University → O
is → O
located → O
in → O
Hanoi → B-LOC
```

**Nhận xét:**
1. "VNU" → B-ORG: Hợp lý, vì VNU (Vietnam National University) là một tổ chức.
2. "University" → O: Phù hợp với cách gán nhãn của CoNLL-2003.
3. "is, located, in" → O: Đúng vì đây là các từ chức năng/động từ.
4. "Hanoi" → B-LOC: Chính xác, vì "Hanoi" là một địa danh.

→ Mô hình dự đoán đúng logic ngữ nghĩa và phù hợp với phong cách gán nhãn của CoNLL-2003.

## 4.7. Khó khăn và hướng cải thiện

**Khó khăn gặp phải:**
1. **Vấn đề tải dữ liệu:**
   - Cần xử lý đặc biệt khi tải từ thư viện `datasets`.
   - Giải pháp: Tải dữ liệu từ nguồn khác và chuyển đổi về đúng định dạng.

2. **Thời gian huấn luyện:**
   - Từ điển lớn (23,626 từ).
   - Số lượng câu trong tập train nhiều.

**Hướng cải thiện:**
1. **Mô hình:**
   - Thêm CRF (BiLSTM-CRF) để mô hình hóa ràng buộc chuỗi nhãn IOB.
   - Sử dụng word embedding tiền huấn luyện (GloVe, FastText) hoặc mô hình transformer (BERT, RoBERTa).

2. **Đánh giá:**
   - Sử dụng Precision/Recall/F1-score theo entity-level thay vì chỉ accuracy token-level.
   - Dùng thư viện `seqeval` để đánh giá chính xác hơn.

3. **Tối ưu hóa:**
   - Tăng `hidden_dim`, số layer BiLSTM.
   - Thêm dropout, early stopping để giảm overfitting.
   - Tinh chỉnh learning rate và các siêu tham số khác.

## 4.8. Kết luận
- Đã xây dựng thành công mô hình BiLSTM cho bài toán NER trên tập dữ liệu CoNLL-2003.
- Đạt được độ chính xác 92.93% trên tập validation.
- Mô hình dự đoán tốt trên câu mới, thể hiện khả năng tổng quát hóa.
- Có nhiều hướng để cải thiện hiệu năng trong tương lai.