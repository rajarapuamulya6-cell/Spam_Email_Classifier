import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Features and labels
X = data['message']
y = data['label']

# Convert text to numbers
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Check accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Test messages continuously
while True:
    user_message = input("\nEnter a message: ")

    test_vector = vectorizer.transform([user_message])
    prediction = model.predict(test_vector)

    if prediction[0] == "spam":
        print("🚨 This is a SPAM message")
    else:
        print("✅ This is a HAM (Normal) message")