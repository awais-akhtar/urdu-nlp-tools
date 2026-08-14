# urdu-nlp-tools

Practical Python utilities for Urdu and Roman Urdu NLP: normalization, preprocessing, transliteration, lightweight augmentation, classification metrics, and sentiment analysis.

The default sentiment analyzer is dependency-free and works immediately with a small Urdu/Roman Urdu lexicon. For real research or production work, train it on your own labeled dataset with the optional scikit-learn extra.

## Install

```bash
pip install urdu-nlp-tools
```

For TF-IDF + LinearSVC training:

```bash
pip install "urdu-nlp-tools[train]"
```

## Quick Start

```python
from urdu_nlp import SentimentAnalyzer

model = SentimentAnalyzer()
print(model.predict("یہ فلم بہت اچھی تھی"))
# positive
```

Batch prediction:

```python
from urdu_nlp import SentimentAnalyzer

model = SentimentAnalyzer()
labels = model.predict([
    "یہ فلم بہت اچھی تھی",
    "movie bohat kharab thi",
])
print(labels)
```

## Train Your Own Sentiment Model

CSV files should include a text column and a label column. By default, the helpers expect `review` and `sentiment`.

```python
from urdu_nlp import SentimentAnalyzer, load_labeled_csv

records = load_labeled_csv("urdu_reviews.csv")
texts = [record.text for record in records]
labels = [record.label for record in records]

model = SentimentAnalyzer().fit(texts, labels)
model.save("urdu_sentiment.pkl")

loaded = SentimentAnalyzer.from_file("urdu_sentiment.pkl")
print(loaded.predict("یہ فلم بہت اچھی تھی"))
```

## Text Utilities

```python
from urdu_nlp import clean_text, normalize_urdu, roman_to_urdu, tokenize_words

print(normalize_urdu("كيا يہ ۱۲۳ ہے؟"))
print(clean_text("@user movie bohat achi thi!!!"))
print(tokenize_words("یہ فلم بہت اچھی تھی"))
print(roman_to_urdu("bohat achi film"))
```

## Metrics

```python
from urdu_nlp import accuracy_score, classification_report

y_true = ["positive", "negative", "neutral"]
y_pred = ["positive", "negative", "positive"]

print(accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))
```

## CLI

```bash
urdu-nlp-sentiment "یہ فلم بہت اچھی تھی"
urdu-nlp-sentiment --model urdu_sentiment.pkl "movie bohat kharab thi"
```

## Package Layout

```text
urdu_nlp/
├── normalize.py
├── sentiment.py
├── transliteration.py
├── augmentation.py
├── preprocessing.py
└── metrics.py
```
