from sklearn.feature_extraction.text import TfidfVectorizer

texts = [
    "This movie is fantastic and I love it!",
    "I hate this film, it's terrible.",
    "The acting was superb, a truly great experience.",
    "What a waste of time, absolutely boring.",
    "Highly recommend this, a masterpiece.",
    "Could not finish watching, so bad."
]
labels = [1, 0, 1, 0, 1, 0]  # 1=Positive, 0=Negative

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(texts)
