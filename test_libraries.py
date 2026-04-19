import torch
import stable_baselines3
import tensorboard
import gymnasium

print(f"PyTorch: {torch.__version__}")
print(f"Stable-Baselines3: {stable_baselines3.__version__}")
print(f"Gymnasium: {gymnasium.__version__}")
print(f"MPS available (Apple Silicon): {torch.backends.mps.is_available()}")
print("All libraries installed correctly!")