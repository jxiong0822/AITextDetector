import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import spacy
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout
from sklearn.preprocessing import LabelEncoder

# Load spaCy language model
nlp = spacy.load("en_core_web_sm")

# Extract structural features from text
def extract_structural_features(texts):
    features = []
    for text in texts:
        doc = nlp(text)
        pos_counts = {pos: 0 for pos in ["NOUN", "VERB", "ADJ", "ADV", "PRON", "PUNCT"]}
        
        for token in doc:
            if token.pos_ in pos_counts:
                pos_counts[token.pos_] += 1

        # Sentence-level features
        sentence_length = len(doc)
        num_sentences = len(list(doc.sents))
        avg_sentence_length = sentence_length / num_sentences if num_sentences > 0 else 0
        
        # Add features as a list
        features.append([
            sentence_length,            # Total number of tokens
            num_sentences,              # Number of sentences
            avg_sentence_length,        # Average sentence length
            pos_counts["NOUN"],         # Number of nouns
            pos_counts["VERB"],         # Number of verbs
            pos_counts["ADJ"],          # Number of adjectives
            pos_counts["ADV"],          # Number of adverbs
            pos_counts["PRON"],         # Number of pronouns
            pos_counts["PUNCT"],        # Number of punctuations
        ])
    return np.array(features)

# Load datasets
def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path, names=["text", "label"], on_bad_lines='skip')
    test_data = pd.read_csv(test_path, names=["text", "label"], on_bad_lines='skip')
    return train_data, test_data

# Train content-based model
def train_content_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Train structural-based model
def train_structural_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# Train LSTM model
def train_lstm_model(texts, labels, max_length=20):
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length)

    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    vocab_size = 10000
    embedding_dim = 128

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(padded_sequences, encoded_labels, epochs=10, batch_size=32, validation_split=0.2)
    return model, tokenizer, encoder

# Predict with LSTM model
def predict_lstm(model, tokenizer, encoder, text, max_length=20):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    prediction = model.predict(padded_sequence)
    return prediction[0][0]

# Main script
def main(train_csv, test_csv):
    # Load train and test datasets
    train_data, test_data = load_data(train_csv, test_csv)
    
    # Extract text and labels
    X_train_raw, y_train = train_data['text'], train_data['label']
    X_test_raw, y_test = test_data['text'], test_data['label']
    
    # Extract content-based features
    vectorizer = CountVectorizer()
    #print("Extracting content-based features...")
    X_train_content = vectorizer.fit_transform(X_train_raw)
    X_test_content = vectorizer.transform(X_test_raw)
    
    # Train content-based model
    #print("Training content-based model...")
    content_model = train_content_model(X_train_content, y_train)
    
    # Extract structural features
    #print("Extracting structural features...")
    X_train_structural = extract_structural_features(X_train_raw)
    X_test_structural = extract_structural_features(X_test_raw)
    
    # Train structural-based model
    #print("Training structural-based model...")
    structural_model = train_structural_model(X_train_structural, y_train)
    
    # Train LSTM model
    #print("Training LSTM model...")
    lstm_model, lstm_tokenizer, lstm_encoder = train_lstm_model(X_train_raw, y_train)

    # Evaluate models
    # content_pred = content_model.predict(X_test_content)
    # structural_pred = structural_model.predict(X_test_structural)
    # lstm_pred = [predict_lstm(lstm_model, lstm_tokenizer, lstm_encoder, text) for text in X_test_raw]

    # print("Test Accuracy (Content-Based):", accuracy_score(y_test, content_pred))
    # print("Test Accuracy (Structural-Based):", accuracy_score(y_test, structural_pred))
    # print("Test Accuracy (LSTM):", accuracy_score(y_test, (np.array(lstm_pred) > 0.5).astype(int)))
    
    # Allow live input
    print("\n=== Live Demo ===")
    print("Enter a text string to predict whether it's AI-generated or not.")
    print("Type 'exit' to quit the live demo.")
    
    while True:
        user_input = input("Your input: ")
        if user_input.lower() == 'exit':
            print("Exiting live demo...")
            break
        
        # Extract features for live input
        content_features = vectorizer.transform([user_input])
        structural_features = extract_structural_features([user_input])
        
        # Get predictions
        content_prob = content_model.predict_proba(content_features)[:, 1]
        structural_prob = structural_model.predict_proba(structural_features)[:, 1]
        lstm_prob = predict_lstm(lstm_model, lstm_tokenizer, lstm_encoder, user_input)
        
        # Combine predictions with equal weight on each
        combined_prob = (2 * content_prob + structural_prob + 2 * lstm_prob) / 5
        final_prediction = (combined_prob > 0.5).astype(int)
        
        print(f"Logistic Regression Prediction: {'AI-generated' if content_prob >= 0.5 else 'Human-written'} (Probability: {content_prob[0]:.2f})")
        print(f"Structural Model Prediction: {'AI-generated' if structural_prob >= 0.5 else 'Human-written'} (Probability: {structural_prob[0]:.2f})")
        print(f"LSTM Model Prediction: {'AI-generated' if lstm_prob >= 0.5 else 'Human-written'} (Probability: {lstm_prob:.2f})")
        print(f"Overall Combined Prediction: {'AI-generated' if final_prediction[0] == 1 else 'Human-written'}")

# Run the script
if __name__ == '__main__':
    # Replace 'train.csv' and 'test.csv' with your actual file paths
    main('data/Train.csv', 'data/Test.csv')
