import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

from .preprocessing import TextPreprocessor
from . import config

class EmotionDetector:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=config.RANDOM_STATE
        )
        
    def train(self, texts, labels):
        """
        Train the emotion detection model
        """
        # Transform texts and labels
        X = self.preprocessor.fit_transform_texts(texts)
        y = self.preprocessor.fit_transform_labels(labels)
        
        # Train the model
        self.model.fit(X, y)
        
        return self
    
    def predict(self, text):
        """
        Predict emotion for a single text input
        """
        # Preprocess the input text
        X = self.preprocessor.preprocess_single_text(text)
        
        # Get prediction
        pred_encoded = self.model.predict(X)
        
        # Convert prediction back to emotion label
        return self.preprocessor.inverse_transform_labels(pred_encoded)[0]
    
    def predict_proba(self, text):
        """
        Get probability distribution across all emotions
        """
        # Preprocess the input text
        X = self.preprocessor.preprocess_single_text(text)
        
        # Get probability distribution
        probas = self.model.predict_proba(X)[0]
        
        # Create dictionary mapping emotions to their probabilities
        emotion_probas = {
            emotion: prob 
            for emotion, prob in zip(self.preprocessor.label_encoder.classes_, probas)
        }
        
        return emotion_probas
    
    def evaluate(self, texts, labels):
        """
        Evaluate model performance on test data
        """
        # Transform texts and labels
        X = self.preprocessor.transform_texts(texts)
        y_true = self.preprocessor.transform_labels(labels)
        
        # Get predictions
        y_pred = self.model.predict(X)
        
        # Convert encoded labels back to emotion labels for readability
        y_true_emotions = self.preprocessor.inverse_transform_labels(y_true)
        y_pred_emotions = self.preprocessor.inverse_transform_labels(y_pred)
        
        return y_true_emotions, y_pred_emotions
    
    def save_model(self, filepath=config.MODEL_SAVE_PATH):
        """
        Save the trained model to disk
        """
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor
        }
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load_model(cls, filepath=config.MODEL_SAVE_PATH):
        """
        Load a trained model from disk
        """
        model_data = joblib.load(filepath)
        
        instance = cls()
        instance.model = model_data['model']
        instance.preprocessor = model_data['preprocessor']
        
        return instance
    
    def get_feature_importance(self, top_n=10):
        """
        Get the most important features (words) for emotion classification
        """
        feature_names = self.preprocessor.get_feature_names()
        importances = self.model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        top_features = []
        for i in range(min(top_n, len(feature_names))):
            top_features.append({
                'feature': feature_names[indices[i]],
                'importance': importances[indices[i]]
            })
        
        return top_features