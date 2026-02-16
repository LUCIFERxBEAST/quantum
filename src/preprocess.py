import pandas as pd
import string

def load_data(filepath):
    """
    Loads the SMS Spam Collection dataset.
    Args:
        filepath (str): Path to the SMSSpamCollection file.
    Returns:
        pd.DataFrame: DataFrame with 'message' and 'label' columns.
                      'label' is mapped to integers (ham=0, spam=1).
    """
    try:
        # The dataset is tab-separated and has no header
        df = pd.read_csv(filepath, sep='\t', header=None, names=['label', 'message'])
        
        # Map labels to binary values
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_text(text):
    """
    Basic text cleaning: lowercase and remove punctuation.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

if __name__ == "__main__":
    # Test loading
    import os
    # Assuming run from project root
    data_path = 'data/SMSSpamCollection'
    if not os.path.exists(data_path):
        # Fallback if running from src
        data_path = '../data/SMSSpamCollection'
        
    df = load_data(data_path)
    if df is not None:
        print("Data loaded successfully:")
        print(df.head())
        print(f"Shape: {df.shape}")
        
        print("\nCleaning example:")
        sample = "Hello!!! This is a TEST..."
        print(f"Original: {sample}")
        print(f"Cleaned: {clean_text(sample)}")
