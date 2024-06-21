

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib  # For saving and loading models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset (assuming it's a CSV with 'Category' and 'Message' columns)
data_path = 'spam.csv'  # Replace with your dataset path
df = pd.read_csv(data_path)

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'\b(?:http|https?://)\S+\b', '', text)
    # Replace specific patterns with unique tokens for improved detection
    patterns = {
        r'\bfree prizes\b': 'free_prizes_token',
        r'\bgift cards\b': 'gift_cards_token',
        r'\bcoupons\b': 'coupons_token',
        r'\baccount verifications\b': 'account_verifications_token',
        r'\btext from government agencies\b': 'govt_agency_token',
        r'\border deliveries\b': 'order_deliveries_token',
        r'\btext from your own number\b': 'own_number_token',
        r'\bcredit card offers\b': 'credit_card_offers_token',
        r'\bunexpected job offers\b': 'unexpected_job_offers_token',
        r'\bissues with your payment information\b': 'payment_issues_token',
        r'\bnoticed suspicious activity\b': 'suspicious_activity_token',
        r'\bfamily emergencies\b': 'family_emergencies_token',
        r'\btwo-factor authentication\b': '2fa_token',
        r'\btexts from your boss\b': 'boss_text_token',
        r'\brefunds and overpayments\b': 'refunds_overpayments_token',
        r'\bsuspicious group texts\b': 'suspicious_group_texts_token'
    }
    for pattern, token in patterns.items():
        text = re.sub(pattern, token, text, flags=re.IGNORECASE)
    return text

# Apply preprocessing to 'Message' column
df['Message'] = df['Message'].apply(preprocess_text)

# Split dataset into X (messages) and y (categories)
X = df['Message']
y = df['Category']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for TF-IDF vectorization and SVM
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', SVC(kernel='linear'))
])

# Train the model
text_clf.fit(X_train, y_train)

# Save the model to a file
model_filename = 'spam_detection_model.pkl'
joblib.dump(text_clf, model_filename)
print(f"Model saved as {model_filename}")

# Evaluation metrics (optional)
y_pred = text_clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
