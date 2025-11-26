import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# Task 1: Đọc và tiền xử lý CoNLL-U


def load_conllu(file_path):
    """
    Đọc file .conllu, trả về:
    [
      [('From', 'ADP'), ('the', 'DET'), ...],
      [('Another', 'DET'), ('sentence', 'NOUN'), ...],
      ...
    ]
    """
    sentences = []
    current_sentence = []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Dòng trống => kết thúc 1 câu
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            # Bỏ qua dòng comment
            if line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 4:
                # dòng lỗi format, bỏ qua
                continue

            token_id = parts[0]
            # Trong CoNLL-U có thể có token kiểu 1-2 hoặc 3.1 => bỏ qua
            if "-" in token_id or "." in token_id:
                continue

            word = parts[1]      # FORM
            upos_tag = parts[3]  # UPOS

            current_sentence.append((word, upos_tag))

        # Câu cuối cùng có thể chưa được push nếu file không kết thúc bằng dòng trống
        if current_sentence:
            sentences.append(current_sentence)

    return sentences



# Task 1: Xây dựng vocabulary


PAD_WORD = "<PAD>"
UNK_WORD = "<UNK>"
PAD_TAG = "<PAD>"   # tag PAD để padding nhãn

def build_vocab(train_sentences):
    """
    Tạo word_to_ix và tag_to_ix từ dữ liệu train.
    Thêm:
      - word_to_ix: PAD_WORD, UNK_WORD
      - tag_to_ix: PAD_TAG
    """
    word_to_ix = {PAD_WORD: 0, UNK_WORD: 1}
    tag_to_ix = {PAD_TAG: 0}

    for sent in train_sentences:
        for word, tag in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

    ix_to_tag = {ix: tag for tag, ix in tag_to_ix.items()}

    return word_to_ix, tag_to_ix, ix_to_tag



# Task 2: Dataset + DataLoader


class POSDataset(Dataset):
    def __init__(self, sentences, word_to_ix, tag_to_ix):
        """
        sentences: list[list[(word, tag)]]
        """
        self.sentences = sentences
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
        self.unk_idx = word_to_ix[UNK_WORD]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sent = self.sentences[idx]
        word_indices = []
        tag_indices = []

        for word, tag in sent:
            word_idx = self.word_to_ix.get(word, self.unk_idx)
            tag_idx = self.tag_to_ix[tag]  # giả sử mọi tag đều có trong vocab
            word_indices.append(word_idx)
            tag_indices.append(tag_idx)

        return (
            torch.tensor(word_indices, dtype=torch.long),
            torch.tensor(tag_indices, dtype=torch.long),
        )


def create_collate_fn(pad_word_idx, pad_tag_idx):
    """
    Hàm tạo collate_fn để dùng cho DataLoader.
    Pad các câu trong batch về cùng độ dài.
    """
    def collate_fn(batch):
        # batch: list of (sentence_tensor, tag_tensor)
        sentences, tags = zip(*batch)  # tuple of tensors

        # pad_sequence: list[tensor(seq_len)] -> tensor(batch, max_len)
        padded_sentences = pad_sequence(
            sentences, batch_first=True, padding_value=pad_word_idx
        )
        padded_tags = pad_sequence(
            tags, batch_first=True, padding_value=pad_tag_idx
        )

        lengths = torch.tensor([len(s) for s in sentences], dtype=torch.long)

        return padded_sentences, padded_tags, lengths

    return collate_fn



# Task 3: Mô hình RNN đơn giản


class SimpleRNNForTokenClassification(nn.Module):
    def __init__(
        self,
        vocab_size,
        tagset_size,
        embedding_dim=100,
        hidden_dim=128,
        padding_idx=0,
        rnn_type="rnn",  # "rnn" hoặc "lstm"
        num_layers=1,
        bidirectional=False,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

        self.rnn_type = rnn_type.lower()
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional
            )
        else:
            self.rnn = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                nonlinearity="tanh",
                bidirectional=bidirectional
            )

        self.fc = nn.Linear(hidden_dim * self.num_directions, tagset_size)

    def forward(self, x):
        """
        x: (batch_size, seq_len) chứa index của từ
        output: logits (batch_size, seq_len, tagset_size)
        """
        # (B, T) -> (B, T, E)
        embedded = self.embedding(x)

        # rnn_out: (B, T, H*num_directions)
        # hidden: (num_layers*num_directions, B, H), (với LSTM thì hidden = (h, c))
        if self.rnn_type == "lstm":
            rnn_out, (h_n, c_n) = self.rnn(embedded)
        else:
            rnn_out, h_n = self.rnn(embedded)

        # (B, T, H*num_directions) -> (B, T, tagset_size)
        logits = self.fc(rnn_out)
        return logits



# Task 4: Huấn luyện


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for sentences, tags, lengths in dataloader:
        sentences = sentences.to(device)
        tags = tags.to(device)

        optimizer.zero_grad()

        # logits: (B, T, C)
        logits = model(sentences)
        B, T, C = logits.shape

        # reshape để dùng CrossEntropyLoss
        logits = logits.view(B * T, C)    # (B*T, C)
        tags = tags.view(B * T)           # (B*T,)

        loss = criterion(logits, tags)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss



# Task 5: Đánh giá


def evaluate(model, dataloader, tag_pad_idx, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for sentences, tags, lengths in dataloader:
            sentences = sentences.to(device)
            tags = tags.to(device)

            logits = model(sentences)           # (B, T, C)
            preds = torch.argmax(logits, dim=-1)  # (B, T)

            # mask để loại bỏ vị trí padding
            mask = (tags != tag_pad_idx)       # (B, T), bool

            correct += (preds[mask] == tags[mask]).sum().item()
            total += mask.sum().item()

    if total == 0:
        return 0.0
    return correct / total


# ==========================
# (Nâng cao) Dự đoán câu mới
# ==========================

def predict_sentence(model, sentence_str, word_to_ix, ix_to_tag, device):
    """
    sentence_str: chuỗi, ví dụ "I love NLP"
    return: list[(word, predicted_tag)]
    """
    model.eval()
    words = sentence_str.strip().split()
    unk_idx = word_to_ix[UNK_WORD]

    idxs = [word_to_ix.get(w, unk_idx) for w in words]
    input_tensor = torch.tensor([idxs], dtype=torch.long).to(device)  # batch size = 1

    with torch.no_grad():
        logits = model(input_tensor)          # (1, T, C)
        preds = torch.argmax(logits, dim=-1)  # (1, T)
        preds = preds.squeeze(0).tolist()     # list length T

    tags = [ix_to_tag[p] for p in preds]
    return list(zip(words, tags))

def main():
    # --------- Đường dẫn dữ liệu (chỉnh lại nếu khác) ----------
    train_path = os.path.join("data", "UD_English-EWT", "en_ewt-ud-train.conllu")
    dev_path = os.path.join("data", "UD_English-EWT", "en_ewt-ud-dev.conllu")

    print("Loading data ...")
    train_sentences = load_conllu(train_path)
    dev_sentences = load_conllu(dev_path)

    print(f"#Train sentences: {len(train_sentences)}")
    print(f"#Dev   sentences: {len(dev_sentences)}")

    # --------- Xây vocab từ train ----------
    word_to_ix, tag_to_ix, ix_to_tag = build_vocab(train_sentences)
    print(f"Vocab size   (words): {len(word_to_ix)}")
    print(f"Vocab size   (tags) : {len(tag_to_ix)}")

    # --------- Dataset & DataLoader ----------
    train_dataset = POSDataset(train_sentences, word_to_ix, tag_to_ix)
    dev_dataset = POSDataset(dev_sentences, word_to_ix, tag_to_ix)

    pad_word_idx = word_to_ix[PAD_WORD]
    pad_tag_idx = tag_to_ix[PAD_TAG]

    collate_fn = create_collate_fn(pad_word_idx, pad_tag_idx)

    batch_size = 32

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # --------- Thiết lập model ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    vocab_size = len(word_to_ix)
    tagset_size = len(tag_to_ix)

    model = SimpleRNNForTokenClassification(
        vocab_size=vocab_size,
        tagset_size=tagset_size,
        embedding_dim=100,
        hidden_dim=128,
        padding_idx=pad_word_idx,
        rnn_type="rnn",      # đổi thành "lstm" nếu muốn LSTM
        num_layers=1,
        bidirectional=False  # có thể True để dùng BiRNN/BiLSTM
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_tag_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --------- Huấn luyện ----------
    n_epochs = 10
    best_dev_acc = 0.0
    best_state_dict = None

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_acc = evaluate(model, train_loader, pad_tag_idx, device)
        dev_acc = evaluate(model, dev_loader, pad_tag_idx, device)

        print(
            f"Epoch {epoch:02d} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_acc*100:.2f}% | "
            f"Dev acc: {dev_acc*100:.2f}%"
        )

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_state_dict = model.state_dict()

    # Load lại model tốt nhất theo dev
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    print(f"\nBest dev accuracy: {best_dev_acc*100:.2f}%")

    # --------- Ví dụ dự đoán câu mới ----------
    example_sentence = "Yesterday I bought a new laptop at the supermarket"
    pairs = predict_sentence(model, example_sentence, word_to_ix, ix_to_tag, device)
    print("\nExample prediction:")
    print("Sentence:", example_sentence)
    print("Prediction:", pairs)


if __name__ == "__main__":
    main()
