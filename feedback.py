import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('feedbacks.csv')

# Display the first few rows of the dataset
print(data.head())

# Split the data into training and test sets
X = data['feedback']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline to vectorize the text and then train a Naive Bayes classifier
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Function to predict sentiment of new feedback
def predict_sentiment(feedback):
    return pipeline.predict([feedback])[0]

# Example usage
new_feedback = "The product is excellent and I'm very satisfied!"
print(f"Feedback: {new_feedback}")
print(f"Predicted Sentiment: {predict_sentiment(new_feedback)}")
