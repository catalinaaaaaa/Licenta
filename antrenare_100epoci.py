import itertools
import random

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
import pandas as pd
from tensorflow.keras.utils import to_categorical

df = pd.read_csv(".\\user_final_script.csv")
df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
df = df.drop('best_model', axis=1)

second_row = df.iloc[1, :]  # criteriul este acuratețea pe lotul de validare
sorted_columns = second_row.sort_values(ascending=False).index
sorted_df = df[sorted_columns]

best_architectures = sorted_df.columns[:5].tolist()

user_paths = []

for model in best_architectures:
    user_paths.append(
        (model+'.h5', model+'.json')
    )
    
sorted_df[sorted_df.columns[:5]]

user_root = '.\\saves\\by_user\\'

# Antrenare pentru user


x_train = np.load(".\data_by_user\XU_train.npy")
x_val = np.load(".\data_by_user\XU_val.npy")
x_test = np.load(".\data_by_user\XU_test.npy")

y_train = np.load(".\data_by_user\YU_train.npy")
y_val = np.load(".\data_by_user\YU_val.npy")
y_test = np.load(".\data_by_user\YU_test.npy")

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

for item in user_paths:
    
    h5_path, json_path = item
    
    json_file = open(user_root + json_path, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(user_root + h5_path)
    
    loaded_model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics="accuracy")
    early_stopping = EarlyStopping(monitor="loss", patience=3)  # 3 epoci ragaz
    
    loaded_model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose = 0
    )
    
    y_prob = loaded_model.predict(x_train)
    y_pred = np.argmax(y_prob, axis=1)
    acc_train = metrics.accuracy_score(np.argmax(y_train,1), y_pred)
    
    y_prob = loaded_model.predict(x_val)
    y_pred = np.argmax(y_prob, axis=1)
    acc_val = metrics.accuracy_score(np.argmax(y_val, 1), y_pred)
    
    y_prob = loaded_model.predict(x_test)
    y_pred = np.argmax(y_prob, axis=1)
    acc_test = metrics.accuracy_score(y_test, y_pred)

    print(f"Pentru modelul {h5_path[:-3]} s-au obținut metricile {acc_train, acc_val, acc_test}")  

#intrauser
df = pd.read_csv(".\\intrauser_final_script.csv")
df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
df = df.drop('best_model', axis=1)

second_row = df.iloc[1, :] # luam dupa validare
sorted_columns = second_row.sort_values(ascending=False).index
sorted_df = df[sorted_columns]

best_architectures = sorted_df.columns[:5].tolist()

intrauser_paths = []

for model in best_architectures:
    intrauser_paths.append(
        (model+'.h5', model+'.json')
    )
    
sorted_df[sorted_df.columns[:5]]

intrauser_root = ".\saves\\by_intrauser\\"

# Antrenare pentru intrauser

x_train = np.load(".\data_by_intrauser\XI_train.npy")
x_val = np.load(".\data_by_intrauser\XI_val.npy")
x_test = np.load(".\data_by_intrauser\XI_test.npy")

y_train = np.load(".\data_by_intrauser\YI_train.npy")
y_val = np.load(".\data_by_intrauser\YI_val.npy")
y_test = np.load(".\data_by_intrauser\YI_test.npy")

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

for item in intrauser_paths:
    
    h5_path, json_path = item
    
    json_file = open(intrauser_root + json_path, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(intrauser_root + h5_path)
    
    loaded_model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics="accuracy")
    early_stopping = EarlyStopping(monitor="loss", patience=3)  # 3 epoci ragaz
    
    loaded_model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose = 0
    )
    
    y_prob = loaded_model.predict(x_train)
    y_pred = np.argmax(y_prob, axis=1)
    acc_train = metrics.accuracy_score(np.argmax(y_train,1), y_pred)
    
    y_prob = loaded_model.predict(x_val)
    y_pred = np.argmax(y_prob, axis=1)
    acc_val = metrics.accuracy_score(np.argmax(y_val, 1), y_pred)
    
    y_prob = loaded_model.predict(x_test)
    y_pred = np.argmax(y_prob, axis=1)
    acc_test = metrics.accuracy_score(y_test, y_pred)

    print(f"Pentru modelul {h5_path[:-3]} s-au obținut metricile {acc_train, acc_val, acc_test}")