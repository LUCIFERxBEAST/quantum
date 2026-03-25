from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class FeatureExtractor:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.pca = PCA(n_components=n_components)
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def fit_transform(self, texts):
        """
        Fits the vectorizer and reducer, then transforms the texts.
        """
        # 1. TF-IDF
        # We limit features to keep PCA efficient
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        # Convert to dense for PCA
        dense_tfidf = tfidf_matrix.toarray()
        
        # 2. PCA (Handle edge cases where n_samples < n_components)
        n_samples, n_feats = dense_tfidf.shape
        if n_samples < self.n_components:
            # PCA cannot reduce 1 sample to 2 dimensions. Pad directly.
            pca_features = np.zeros((n_samples, self.n_components))
            cols = min(n_feats, self.n_components)
            pca_features[:, :cols] = dense_tfidf[:, :cols]
        else:
            pca_features = self.pca.fit_transform(dense_tfidf)
        
        # 3. Scaling for Quantum
        # Quantum gates usually rotate by angle * feature, so [0, 1] mapped to [0, 2pi] 
        # or just keeping it [0,1] is often fine if we handle the scaling in the circuit.
        # Here we map to [0, 1]
        scaled_features = self.scaler.fit_transform(pca_features)
        
        return scaled_features

    def transform(self, texts):
        """
        Transforms new texts using the fitted pipeline.
        """
        tfidf_matrix = self.vectorizer.transform(texts)
        dense_tfidf = tfidf_matrix.toarray()
        pca_features = self.pca.transform(dense_tfidf)
        scaled_features = self.scaler.transform(pca_features)
        return scaled_features

if __name__ == "__main__":
    # Simple test
    texts = [
        "Win a free prize", 
        "Meeting at 5pm", 
        "Get cheap loans", 
        "Lunch tomorrow", 
        "Free money now", 
        "Project status update"
    ]
    fe = FeatureExtractor(n_components=2)
    features = fe.fit_transform(texts)
    print("Features shape:", features.shape)
    print("Features:\n", features)
