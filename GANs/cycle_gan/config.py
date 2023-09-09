import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths for saving/loading models
CHECKPOINT_GEN = "gen.pth.tar"
CHECKPOINT_DISC = "disc.pth.tar"

# Training parameters
BATCH_SIZE = 64
LEARNING_RATE = 2e-4  # Commonly used value for Adam optimizer with GANs
NOISE_DIM = 100  # Length of noise vector for generator
# Number of features in your dataset (this needs to be updated based on your dataset)
NUM_FEATURES = 5
NUM_EPOCHS = 50
NUM_SAMPLES = 1000
# Loss weights (if applicable)
LAMBDA = 10  # Weight for any additional losses you might want to use

# Control flags
LOAD_MODEL = False
SAVE_MODEL = True

# For reproducibility
SEED = 42

# If you have specific paths for training/validation datasets
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
