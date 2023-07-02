import itertools
import random
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import clear_output
from sklearn import metrics
from tensorflow.keras import Sequential
from tensorflow.keras.activations import sigmoid, softmax
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

x_train = np.load("./data_by_user/XU_train.npy")
x_val = np.load("./data_by_user/XU_val.npy")
x_test = np.load("./data_by_user/XU_test.npy")

y_train = np.load("./data_by_user/YU_train.npy")
y_val = np.load("./data_by_user/YU_val.npy")
y_test = np.load("./data_by_user/YU_test.npy")

files = []

folder_path = "./saves/by_user"

file_pattern = os.path.join(folder_path, "*.h5")

for file_path in glob.glob(file_pattern):
    if os.path.isfile(file_path):
        fixed_path = file_path.replace("\\", "/")
        files.append((fixed_path, fixed_path[:-2] + "json"))
        
accuracy_dict = {}

for i, item in enumerate(files):

    h5_path, json_path = item

    with open(json_path, "r") as json_file:
        loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(h5_path)

    y_prob = loaded_model.predict(x_train)
    y_pred = np.argmax(y_prob, axis=1)
    accuracy_train = metrics.accuracy_score(y_train, y_pred)

    y_prob = loaded_model.predict(x_val)
    y_pred = np.argmax(y_prob, axis=1)
    accuracy_val = metrics.accuracy_score(y_val, y_pred)

    y_prob = loaded_model.predict(x_test)
    y_pred = np.argmax(y_prob, axis=1)
    accuracy_test = metrics.accuracy_score(y_test, y_pred)

    accuracy_dict[h5_path[16:-3]] = [accuracy_train, accuracy_val, accuracy_test]

    print(f"S-a finalizat cu succes pasul {i+1}/6801. Ne aflam la pasul {h5_path[16:-3]}")
    clear_output(wait=True)
    
df = pd.DataFrame(accuracy_dict)
df.to_csv("user_final_script.csv")