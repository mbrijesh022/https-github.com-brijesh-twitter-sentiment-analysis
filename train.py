import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

nltk.download('stopwords')
stop_words = stopwords.words('english')

# ── Load Sentiment140 dataset ──────────────────────────────────────────────
# Download from: https://www.kaggle.com/datasets/kazanova/sentiment140
# Save as 'training.1600000.processed.noemoticon.csv' in the same folder

print("Loading dataset...")
df = pd.read_csv(
    'training.1600000.processed.noemoticon.csv',
    encoding='latin-1',
    header=None,
    names=['target', 'id', 'date', 'flag', 'user', 'text']
)

# Keep only text and label; map 4 → 1 (positive)
df = df[['target', 'text']]
df['target'] = df['target'].map({0: 0, 4: 1})

# Use a 200,000 sample for faster training (100k each class)
df = df.groupby('target').sample(n=100000, random_state=42).reset_index(drop=True)

# ── Preprocessing ──────────────────────────────────────────────────────────
def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [w for w in text if w not in stop_words]
    return ' '.join(text)

print("Preprocessing tweets...")
df['clean'] = df['text'].apply(preprocess)

# ── TF-IDF Vectorization ───────────────────────────────────────────────────
print("Vectorizing...")
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['clean'])
y = df['target']

# ── Train Linear SVM ───────────────────────────────────────────────────────
print("Training model...")
model = LinearSVC(C=1.0, max_iter=2000)
model.fit(X, y)

# ── Save model and vectorizer ──────────────────────────────────────────────
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Done! model.pkl and vectorizer.pkl saved successfully.")
