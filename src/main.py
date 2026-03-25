import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

from preprocess import load_data, clean_text
from features import FeatureExtractor
from quantum_model import QuantumSpamClassifier
from grover_classifier import GroverSpamClassifier

def main():
    parser = argparse.ArgumentParser(description="Quantum Accelerated Spam Mail Checker")
    parser.add_argument('--data', type=str, default='data/SMSSpamCollection', help='Path to dataset')
    parser.add_argument('--samples', type=int, default=200, help='Number of samples to use (subset)')
    parser.add_argument('--qubits', type=int, default=2, help='Number of qubits/features')
    parser.add_argument('--test_size', type=float, default=0.3, help='Test set size ratio')
    parser.add_argument('--classical', action='store_true', help='Use classical SVM instead of Quantum')
    parser.add_argument('--grover', action='store_true', help='Use Grover-based Search Classifier')
    
    args = parser.parse_args()
    
    # 1. Load Data
    print(f"Loading data from {args.data}...")
    import os
    if not os.path.exists(args.data):
        # Allow running from src folder
        if os.path.exists(f"../{args.data}"):
            args.data = f"../{args.data}"
            
    df = load_data(args.data)
    if df is None:
        print("Failed to load data.")
        return

    # 2. Preprocess
    print("Cleaning text...")
    df['clean_message'] = df['message'].apply(clean_text)
    
    # Subsample for simulation speed
    # We balance the dataset if possible to make the small sample representative
    spam = df[df['label'] == 1]
    ham = df[df['label'] == 0]
    
    if args.samples > 0 and args.samples < len(df):
        print(f"Subsampling to {args.samples} samples...")
        # Try to keep some balance
        n_spam = int(args.samples / 2)
        n_ham = args.samples - n_spam
        
        # Taking random samples
        spam_subset = spam.sample(n=min(len(spam), n_spam), random_state=42)
        ham_subset = ham.sample(n=min(len(ham), n_ham), random_state=42)
        
        df = pd.concat([spam_subset, ham_subset])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Dataset size: {len(df)}")
    print(df['label'].value_counts())

    X_text = df['clean_message'].values
    y = df['label'].values
    
    # 3. Feature Extraction
    print(f"Extracting features (TF-IDF -> PCA -> {args.qubits} dims)...")
    fe = FeatureExtractor(n_components=args.qubits)
    X = fe.fit_transform(X_text)
    
    # 4. Train/Test Split (Only for Learning Models)
    if not args.grover:
        # Standard ML Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)
        
        use_quantum = not args.classical
        if use_quantum:
            model = QuantumSpamClassifier(n_qubits=args.qubits, use_quantum=True)
        else:
            print("Initializing Classical SVM...")
            model = QuantumSpamClassifier(n_qubits=args.qubits, use_quantum=False) # Helper wrapper

        model.train(X_train, y_train)
        
        # 6. Evaluation
        print("Evaluating model...")
        y_pred = model.predict(X_test)
        
        print("\n--- Classification Report ---")
        print(classification_report(y_test, y_pred, labels=[0, 1], target_names=['Ham', 'Spam'], zero_division=0))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    else:
        # 5. Grover's "Classification" (Search)
        # Grover doesn't learn, so we don't need a split. We can run it on the FULL loaded dataset.
        print(f"Running Grover's Search on ALL {len(X_text)} loaded samples...")
        
        # Expanded list of suspicious keywords for a "Smarter" Oracle
        # Source: User provided categories
        keywords = [
            # Financial/Urgency
            'act', 'now', 'immediate', 'limited', 'time', 'prize', 'claims', 'cash', 'bonus',
            # Marketing/Sales
            'buy', 'clearance', 'discount', 'offer', 'promo', 'subscribe', 'trial',
            # Account/Security
            'verify', 'suspend', 'security', 'unauthorized', 'password', 'login',
            # Common "Spammy" verbs
            'earn', 'save', 'guaranteed', 'exclusive', 'congratulations',
            # Original High-Frequency Spam Words
            'win', 'free', 'urgent', 'call', 'winner', 'selected', 'mobile', 'text', 'stop', 'reply'
        ]
        print(f"Using Grover-based Classifier (Keywords: {keywords})...")
        model = GroverSpamClassifier(suspicious_keywords=keywords)
        
        print("Evaluating model...")
        # Run on EVERYTHING
        y_pred = model.predict(X_text)
        
        print("\n--- Grover's Algorithm Classification Report (Full Dataset) ---")
        print(classification_report(y, y_pred, labels=[0, 1], target_names=['Ham', 'Spam'], zero_division=0))
        print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")

if __name__ == "__main__":
    main()
