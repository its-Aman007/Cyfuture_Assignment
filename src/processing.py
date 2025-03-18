import re
import neattext.functions as nfx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

class TextPreprocessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        self.label_encoder = LabelEncoder()
        
    def clean_text(self, text):
        """
        Clean and normalize text data
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = nfx.remove_special_characters(text)
        text = nfx.remove_numbers(text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove URLs
        text = nfx.remove_urls(text)
        
        # Remove email addresses
        text = nfx.remove_emails(text)
        
        # Remove common punctuation
        text = nfx.remove_punctuations(text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def fit_transform_texts(self, texts):
        """
        Fit and transform text data using TF-IDF vectorization
        """
        # Clean each text
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Fit and transform using TF-IDF vectorizer
        return self.vectorizer.fit_transform(cleaned_texts)
    
    def transform_texts(self, texts):
        """
        Transform text data using fitted TF-IDF vectorizer
        """
        # Clean each text
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Transform using fitted vectorizer
        return self.vectorizer.transform(cleaned_texts)
    
    def fit_transform_labels(self, labels):
        """
        Fit and transform emotion labels
        """
        return self.label_encoder.fit_transform(labels)
    
    def transform_labels(self, labels):
        """
        Transform emotion labels using fitted encoder
        """
        return self.label_encoder.transform(labels)
    
    def inverse_transform_labels(self, encoded_labels):
        """
        Convert encoded labels back to original emotion labels
        """
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def get_feature_names(self):
        """
        Get feature names (words) from the vectorizer
        """
        return self.vectorizer.get_feature_names_out()

    def preprocess_single_text(self, text):
        """
        Preprocess a single text input for prediction
        """
        cleaned_text = self.clean_text(text)
        return self.vectorizer.transform([cleaned_text])