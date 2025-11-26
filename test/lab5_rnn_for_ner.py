import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset



# Config chung


PAD_WORD = "<PAD>"
UNK_WORD = "<UNK>"
PAD_TAG = "<PAD>"  # tag padding



# Task 1: Tải và tiền xử lý CoNLL 2003


def load_conll2003():
    """
    Tải bộ dữ liệu CoNLL-2003 từ Hugging Face.
    Trả về:
      train_sentences, train_tags_str, val_sentences, val_tags_str, label_names
    Trong đó:
      - sentences: list[list[str]]
      - tags_str: list[list[str]] (B-PER, I-ORG, O, ...)
    """
    print("Loading CoNLL-2003 from Hugging Face...")
    dataset = load_dataset("conll2003", trust_remote_code=True)

    # Danh sách tên nhãn (string) cho ner_tags
    label_names = dataset["train"].features["ner_tags"].feature.names
    # Ví dụ: ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

    def convert_split(split_name):
        tokens_list = dataset[split_name]["tokens"]      # list[list[str]]
        tags_id_list = dataset[split_name]["ner_tags"]   # list[list[int]]
        # Chuyển id -> string
        tags_str_list = [
            [label_names[tag_id] for tag_id in seq] for seq in tags_id_list
        ]
        return tokens_list, tags_str_list

    train_sentences, train_tags_str = convert_split("train")
    val_sentences, val_tags_str = convert_split("validation")

    print(f"#Train sentences: {len(train_sentences)}")
    print(f"#Val   sentences: {len(val_sentences)}")

    return train_sentences, train_tags_str, val_sentences, val_tags_str, label_names



# Task 1: Xây vocabulary cho từ và nhãn


def build_vocab(train_sentences, train_tags_str, label_names):
    """
    Xây dựng:
      - word_to_ix (có <PAD>, <UNK>)
      - tag_to_ix (có <PAD>)
      - ix_to_tag
    """
    # word vocab
    word_to_ix = {PAD_WORD: 0, UNK_WORD: 1}
    for sent in train_sentences:
        for w in sent:
            if w not in word_to_ix:
                word_to_ix[w] = len(word_to_ix)

    # tag vocab
    tag_to_ix = {PAD_TAG: 0}
    # label_names chứa các tag như 'O', 'B-PER', ...
    for tag in label_names:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)

    ix_to_tag = {ix: tag for tag, ix in tag_to_ix.items()}

    print(f"Vocab size (words): {len(word_to_ix)}")
    print(f"Vocab size (tags) : {len(tag_to_ix)}")

    return word_to_ix, tag_to_ix, ix_to_tag



# Task 2: Dataset + DataLoader


class NERDataset(Dataset):
    def __init__(self, sentences, tags_str, word_to_ix, tag_to_ix):
        """
        sentences: list[list[str]]
        tags_str:  list[list[str]]  (B-PER, I-ORG, O, ...)
        """
        assert len(sentences) == len(tags_str)
        self.sentences = sentences
        self.tags_str = tags_str
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
        self.unk_idx = word_to_ix[UNK_WORD]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words = self.sentences[idx]
        tags = self.tags_str[idx]

        # Chuyển từ -> index
        word_indices = [
            self.word_to_ix.get(w, self.unk_idx) for w in words
        ]
        # Chuyển nhãn -> index
        tag_indices = [
            self.tag_to_ix[t] for t in tags
        ]

        return (
            torch.tensor(word_indices, dtype=torch.long),
            torch.tensor(tag_indices, dtype=torch.long),
        )


def create_collate_fn(pad_word_idx, pad_tag_idx):
    """
    collate_fn để padding câu trong batch
    """
    def collate_fn(batch):
        sentences, tags = zip(*batch)  # tuple of tensors

        padded_sentences = pad_sequence(
            sentences, batch_first=True, padding_value=pad_word_idx
        )
        padded_tags = pad_sequence(
            tags, batch_first=True, padding_value=pad_tag_idx
        )

        lengths = torch.tensor([len(s) for s in sentences], dtype=torch.long)
        return padded_sentences, padded_tags, lengths

    return collate_fn



# Task 3: BiLSTM cho token classification


class BiLSTMForTokenClassification(nn.Module):
    def __init__(
        self,
        vocab_size,
        tagset_size,
        embedding_dim=100,
        hidden_dim=128,
        padding_idx=0,
        num_layers=1,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # BiLSTM => hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, x):
        """
        x: (B, T)  (indices của từ)
        -> logits: (B, T, tagset_size)
        """
        embedded = self.embedding(x)        # (B, T, E)
        lstm_out, _ = self.lstm(embedded)   # (B, T, 2H)
        logits = self.fc(lstm_out)          # (B, T, C)
        return logits



# Task 4: Huấn luyện


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for sentences, tags, lengths in dataloader:
        sentences = sentences.to(device)
        tags = tags.to(device)

        optimizer.zero_grad()

        logits = model(sentences)  # (B, T, C)
        B, T, C = logits.shape

        logits = logits.view(B * T, C)
        tags = tags.view(B * T)

        loss = criterion(logits, tags)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


# ==========================
# Task 5: Đánh giá
# ==========================

def evaluate(model, dataloader, pad_tag_idx, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for sentences, tags, lengths in dataloader:
            sentences = sentences.to(device)
            tags = tags.to(device)

            logits = model(sentences)        # (B, T, C)
            preds = torch.argmax(logits, dim=-1)  # (B, T)

            # mask để bỏ padding
            mask = (tags != pad_tag_idx)

            correct += (preds[mask] == tags[mask]).sum().item()
            total += mask.sum().item()

    if total == 0:
        return 0.0
    return correct / total



# Dự đoán câu mới


def predict_sentence(model, sentence_str, word_to_ix, ix_to_tag, device):
    """
    sentence_str: chuỗi câu, ví dụ:
      "VNU University is located in Hanoi"
    Trả về: list[(token, predicted_ner_tag)]
    """
    model.eval()
    tokens = sentence_str.strip().split()  # tokenization đơn giản
    unk_idx = word_to_ix[UNK_WORD]

    idxs = [word_to_ix.get(tok, unk_idx) for tok in tokens]
    input_tensor = torch.tensor([idxs], dtype=torch.long).to(device)  # (1, T)

    with torch.no_grad():
        logits = model(input_tensor)              # (1, T, C)
        preds = torch.argmax(logits, dim=-1)      # (1, T)
        preds = preds.squeeze(0).tolist()         # list length T

    tags = [ix_to_tag[p] for p in preds]
    return list(zip(tokens, tags))


def main():
    # ----- Task 1: Tải & xử lý dữ liệu -----
    train_sentences, train_tags_str, val_sentences, val_tags_str, label_names = load_conll2003()

    # ----- Xây vocab -----
    word_to_ix, tag_to_ix, ix_to_tag = build_vocab(
        train_sentences, train_tags_str, label_names
    )

    pad_word_idx = word_to_ix[PAD_WORD]
    pad_tag_idx = tag_to_ix[PAD_TAG]

    # ----- Dataset & DataLoader -----
    train_dataset = NERDataset(train_sentences, train_tags_str, word_to_ix, tag_to_ix)
    val_dataset = NERDataset(val_sentences, val_tags_str, word_to_ix, tag_to_ix)

    collate_fn = create_collate_fn(pad_word_idx, pad_tag_idx)

    batch_size = 32
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # ----- Mô hình BiLSTM -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    vocab_size = len(word_to_ix)
    tagset_size = len(tag_to_ix)

    model = BiLSTMForTokenClassification(
        vocab_size=vocab_size,
        tagset_size=tagset_size,
        embedding_dim=100,
        hidden_dim=128,
        padding_idx=pad_word_idx,
        num_layers=1,
        dropout=0.1,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_tag_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ----- Huấn luyện -----
    n_epochs = 5
    best_val_acc = 0.0
    best_state_dict = None

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, pad_tag_idx, device)

        print(
            f"Epoch {epoch}/{n_epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val acc: {val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict()

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    print(f"\nBest validation accuracy: {best_val_acc*100:.2f}%")

    # ----- Ví dụ dự đoán câu mới -----
    example_sentence = "VNU University is located in Hanoi"
    pairs = predict_sentence(model, example_sentence, word_to_ix, ix_to_tag, device)

    print("\nExample prediction:")
    print("Sentence:", example_sentence)
    for tok, tag in pairs:
        print(f"{tok:15s} -> {tag}")


if __name__ == "__main__":
    main()
