import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from . import config

def ensure_directories_exist():
    """
    Create necessary directories if they don't exist
    """
    directories = [
        config.DATA_DIR,
        config.MODELS_DIR,
        config.TRAIN_DATA_PATH,
        config.VAL_DATA_PATH
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def load_data(data_path):
    """
    Load text data from directories where each subdirectory name is the emotion label
    Returns:
        pandas DataFrame with 'text' and 'emotion' columns
    """
    texts = []
    labels = []
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    for emotion in config.EMOTION_LABELS:
        emotion_path = os.path.join(data_path, emotion)
        if not os.path.exists(emotion_path):
            continue
        
        for file_name in os.listdir(emotion_path):
            if file_name.endswith('.txt'):
                try:
                    with open(os.path.join(emotion_path, file_name), 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if text:  # Only add non-empty texts
                            texts.append(text)
                            labels.append(emotion)
                except Exception as e:
                    print(f"Error reading file {file_name}: {str(e)}")
    
    return pd.DataFrame({'text': texts, 'emotion': labels})

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix for model evaluation
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create heatmap
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        xticklabels=config.EMOTION_LABELS,
        yticklabels=config.EMOTION_LABELS
    )
    
    plt.title('Confusion Matrix (%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_emotion_distribution(labels, save_path=None):
    """
    Plot distribution of emotions in the dataset
    Args:
        labels: List of emotion labels
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Count emotions
    emotion_counts = pd.Series(labels).value_counts()
    
    # Create bar plot with custom colors
    bars = plt.bar(emotion_counts.index, emotion_counts.values)
    
    # Color each bar according to emotion
    for bar, emotion in zip(bars, emotion_counts.index):
        bar.set_color(config.EMOTION_COLORS[emotion])
    
    plt.title('Distribution of Emotions in Dataset')
    plt.xlabel('Emotions')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Add count labels on top of each bar
    for i, v in enumerate(emotion_counts.values):
        plt.text(i, v, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def print_classification_metrics(y_true, y_pred):
    """
    Print detailed classification metrics
    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    # Calculate and print accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    report = classification_report(
        y_true,
        y_pred,
        target_names=config.EMOTION_LABELS,
        digits=4
    )
    print(report)

def create_emotion_probability_plot(probabilities, save_path=None):
    """
    Create bar plot of emotion probabilities
    Args:
        probabilities: Dictionary mapping emotions to their probabilities
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    emotions = list(probabilities.keys())
    probs = list(probabilities.values())
    
    # Create bars
    bars = plt.bar(emotions, probs)
    
    # Color bars according to emotions
    for bar, emotion in zip(bars, emotions):
        bar.set_color(config.EMOTION_COLORS[emotion])
    
    plt.title('Emotion Probability Distribution')
    plt.xlabel('Emotions')
    plt.ylabel('Probability')
    plt.xticks(rotation=45)
    
    # Add probability values on top of each bar
    for i, prob in enumerate(probs):
        plt.text(i, prob, f'{prob:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        return plt

def validate_data_directory(directory):
    """
    Validate that the data directory has the correct structure
    Args:
        directory: Path to data directory
    Returns:
        bool: True if valid, raises Exception if invalid
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Check for emotion subdirectories
    found_emotions = set(os.listdir(directory))
    required_emotions = set(config.EMOTION_LABELS)
    
    missing_emotions = required_emotions - found_emotions
    if missing_emotions:
        raise ValueError(
            f"Missing emotion directories in {directory}: {missing_emotions}"
        )
    
    # Check for text files in each emotion directory
    for emotion in config.EMOTION_LABELS:
        emotion_dir = os.path.join(directory, emotion)
        files = os.listdir(emotion_dir)
        text_files = [f for f in files if f.endswith('.txt')]
        
        if not text_files:
            raise ValueError(
                f"No text files found in {emotion_dir}"
            )
    
    return True