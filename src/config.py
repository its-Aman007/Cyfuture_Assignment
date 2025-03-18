import os

# Project directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Data paths
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train')
VAL_DATA_PATH = os.path.join(DATA_DIR, 'validation')
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'emotion_detector.joblib')

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 5000  # For TF-IDF vectorizer

# Training parameters
BATCH_SIZE = 32

# GUI settings
WINDOW_TITLE = "Emotion Detection System"
WINDOW_SIZE = "800x600"
FONT_FAMILY = "Arial"
FONT_SIZE_TITLE = 14
FONT_SIZE_TEXT = 12
BG_COLOR = "#f0f0f0"
BUTTON_COLOR = "#4CAF50"
TEXT_COLOR = "#000000"

# Emotion labels
EMOTION_LABELS = [
    'angry',
    'disgust',
    'fear',
    'happy',
    'neutral',
    'sad',
    'surprise'
]

# Colors for emotion visualization
EMOTION_COLORS = {
    'angry': '#ff0000',      # Red
    'disgust': '#800080',    # Purple
    'fear': '#808080',       # Gray
    'happy': '#ffff00',      # Yellow
    'neutral': '#0000ff',    # Blue
    'sad': '#964b00',        # Brown
    'surprise': '#ffa500'    # Orange
}

# Model hyperparameters
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}

# Text preprocessing parameters
TEXT_PREPROCESSING = {
    'remove_urls': True,
    'remove_emails': True,
    'remove_numbers': True,
    'remove_special_chars': True,
    'remove_punctuations': True,
    'min_word_length': 2
}