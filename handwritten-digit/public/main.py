import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
len(X_train)
plt.matshow(X_train[0])
print("hi")