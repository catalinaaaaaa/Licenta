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
from tensorflow.keras.utils import to_categorical

def create_mlp(n_hidden_layers, n_neurons_list, function):

    list_of_models = []

    n_neurons = list(itertools.product(n_neurons_list, repeat=n_hidden_layers))

    for combination in n_neurons:
        model = Sequential()
        model.add(Input(shape=(15,)))
        for n_units in combination:
            model.add(Dense(units=n_units, activation=function))

        model.add(Dense(units=7, activation=softmax))
        model.compile(
            optimizer=Adam(), loss="categorical_crossentropy", metrics="accuracy"
        )
        list_of_models.append(model)

    return list_of_models

x_train = np.load("./data_by_intrauser/XI_train.npy")
x_val = np.load("./data_by_intrauser/XI_val.npy")
x_test = np.load("./data_by_intrauser/XI_test.npy")

y_train = np.load("./data_by_intrauser/YI_train.npy")
y_val = np.load("./data_by_intrauser/YI_val.npy")
y_test = np.load("./data_by_intrauser/YI_test.npy")

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

functions_list = ["linear", "relu", "selu", "tanh", "sigmoid"]
n_hidden_layers_list = [2, 3, 4, 5]
n_neurons_list = [8, 16, 32, 64]

hyperparameters_list = list(itertools.product(functions_list, n_hidden_layers_list))

best_model = None
best_accuracy = 0
iteration_index = 0

# 6800 = (4**2 + 4**3 + 4**4 + 4**5) * 5

for hyperparameters in hyperparameters_list:
    function = hyperparameters[0]
    n_hidden_layers = hyperparameters[1]
    list_of_models = create_mlp(n_hidden_layers, n_neurons_list, function)

    for i, model in enumerate(list_of_models):

        early_stopping = EarlyStopping(monitor="loss", patience=3)

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=1,
            batch_size=32,
            verbose=0,
            callbacks=[early_stopping],
        )

        y_prob = model.predict(x_test)
        y_pred = np.argmax(y_prob, axis=1)

        if metrics.accuracy_score(y_test, y_pred) > best_accuracy:
            best_accuracy = metrics.accuracy_score(y_test, y_pred)
            best_model = model

        model_json = model.to_json()
        with open(
            f"./saves/by_intrauser/model_{function}_{n_hidden_layers}_{i+1}.json", "w"
        ) as json_file:
            json_file.write(model_json)
        model.save_weights(
            f"./saves/by_intrauser/model_{function}_{n_hidden_layers}_{i+1}.h5"
        )

        iteration_index += 1

        print(
            f"Suntem la combinatia {function, n_hidden_layers}. S-a terminat pasul {iteration_index}/6800"
        )
        clear_output(wait=True)

model_json = best_model.to_json()
with open("./saves/by_intrauser/best_model.json", "w") as json_file:
    json_file.write(model_json)

best_model.save_weights("./saves/by_intrauser/best_model.h5")