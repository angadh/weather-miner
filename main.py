"""
This is the main file for the machine learning application.
"""
import torch
import torchvision
import tensorflow as tf

if __name__ == "__main__":
    print("Torch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("TensorFlow version:", tf.__version__)
    print("Keras version:", tf.keras.__version__)
