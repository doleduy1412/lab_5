from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from src.models.text_classifier import TextClassifier

texts = [
    "This movie is fantastic and I love it!",
    "I hate this film, it's terrible.",
    "The acting was superb, a truly great experience.",
    "What a waste of time, absolutely boring.",
    "Highly recommend this, a masterpiece.",
    "Could not finish watching, so bad."
]
labels = [1, 0, 1, 0, 1, 0]

# Split data
train_texts, test_texts, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Model
clf = TextClassifier(vectorizer)
clf.fit(train_texts, y_train)
y_pred = clf.predict(test_texts)

# Evaluate
metrics = clf.evaluate(y_test, y_pred)
print("Evaluation metrics:", metrics)
