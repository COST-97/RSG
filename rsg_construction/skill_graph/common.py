import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCORE_FACTOR = 3
# DEVICE = "cpu"