import os
import argparse
from sklearn.model_selection import train_test_split

from .model import EmotionDetector
from .utils import (
    load_data,
    plot_confusion_matrix,
    plot_emotion_distribution,
    print_classification_metrics,
    ensure_directories_exist
)
from . import config

def train_model(train_data_path=config.TRAIN_DATA_PATH, val_data_path=config.VAL_DATA_PATH):
    """
    Train the emotion detection model using the provided data
    """
    # Ensure required directories exist
    ensure_directories_exist()
    
    print("Loading training data...")
    train_df = load_data(train_data_path)
    
    print("Loading validation data...")
    val_df = load_data(val_data_path)
    
    # Plot emotion distribution
    print("Plotting emotion distribution...")
    plot_emotion_distribution(train_df['emotion'])
    
    # Initialize and train the model
    print("Training model...")
    model = EmotionDetector()
    model.train(train_df['text'], train_df['emotion'])
    
    # Evaluate on validation set
    print("\nEvaluating model on validation set...")
    y_true, y_pred = model.evaluate(val_df['text'], val_df['emotion'])
    
    # Print classification metrics
    print_classification_metrics(y_true, y_pred)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred)
    
    # Get and print feature importance
    print("\nTop important features for emotion detection:")
    top_features = model.get_feature_importance(top_n=10)
    for feat in top_features:
        print(f"Feature: {feat['feature']}, Importance: {feat['importance']:.4f}")
    
    # Save the model
    print("\nSaving model...")
    model.save_model()
    print(f"Model saved to {config.MODEL_SAVE_PATH}")
    
    return model

def test_model_predictions(model, texts):
    """
    Test model predictions on sample texts
    """
    print("\nTesting model predictions:")
    for text in texts:
        emotion = model.predict(text)
        probas = model.predict_proba(text)
        
        print(f"\nText: {text}")
        print(f"Predicted emotion: {emotion}")
        print("Emotion probabilities:")
        for emotion, prob in sorted(probas.items(), key=lambda x: x[1], reverse=True):
            print(f"{emotion}: {prob:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Train emotion detection model')
    parser.add_argument('--train_path', type=str, default=config.TRAIN_DATA_PATH,
                        help='Path to training data directory')
    parser.add_argument('--val_path', type=str, default=config.VAL_DATA_PATH,
                        help='Path to validation data directory')
    
    args = parser.parse_args()
    
    # Train the model
    model = train_model(args.train_path, args.val_path)
    
    # Test the model on some sample texts
    sample_texts = [
        "I'm so happy today!",
        "This makes me really angry.",
        "I'm feeling a bit sad right now.",
        "Wow, this is such a surprise!",
        "I'm quite neutral about this situation."
    ]
    
    test_model_predictions(model, sample_texts)

if __name__ == "__main__":
    main()