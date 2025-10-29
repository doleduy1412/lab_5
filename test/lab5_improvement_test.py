from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re

# --- 1. Clean function (advanced preprocessing) ---
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

# --- 2. Dataset ---
texts = [
    "This movie is fantastic and I love it!",
    "I hate this film, it's terrible.",
    "The acting was superb, a truly great experience.",
    "What a waste of time, absolutely boring.",
    "Highly recommend this, a masterpiece.",
    "Could not finish watching, so bad."
]
labels = [1, 0, 1, 0, 1, 0]
texts = [clean_text(t) for t in texts]

# --- 3. Split ---
train_texts, test_texts, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# --- 4. Vectorize ---
vectorizer = TfidfVectorizer(stop_words="english")
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# --- 5. Train Naive Bayes (Improved model) ---
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- 6. Evaluate ---
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
}
print("Improved model (Naive Bayes) metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.3f}")
