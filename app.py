

import streamlit as st
import joblib
import re

# Load the saved model
model_filename = 'spam_detection_model.pkl'
loaded_model = joblib.load(model_filename)

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

# Function to predict whether a message is 'ham' (safe) or 'spam'
def predict_message(message):
    # Preprocess the new message
    cleaned_message = preprocess_text(message)
    # Use the loaded model to predict
    prediction = loaded_model.predict([cleaned_message])[0]
    return prediction

# Streamlit UI
def main():
    st.title("Spam Detection App")
    st.write("Enter a message to determine if it's safe (ham) or spam.")

    # Input text box for user to enter a message
    message = st.text_area("Enter message here:")

    # Button to classify message
    if st.button("Classify"):
        if message:
            prediction = predict_message(message)
            st.write(f"Prediction: {prediction}")
        else:
            st.write("Please enter a message.")

if __name__ == "__main__":
    main()
