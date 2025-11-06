#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project 2 

@author: nekaumakanth
"""

 #1.0 Data Processing 
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory 


np.random.seed(42)
keras.utils.set_random_seed(42)

base_directory = "Data"
IMG_SIZE = (500,500)
BATCH_SIZE = 32

# Loading train, test and validation sets
train_data = image_dataset_from_directory(
    base_directory + "/train",
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    # one hot encoding form, categorical data
    label_mode = "categorical"
    )

test_data = image_dataset_from_directory(
    base_directory + "/test",
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    label_mode = "categorical",
    shuffle = False #needed for data evalutation 
    )

val_data = image_dataset_from_directory(
    base_directory + "/valid",
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    label_mode = "categorical"
    )

class_names = train_data.class_names
print("Classes:", class_names)
 
# Data augmentation only on training set 
data_augmentation = keras.Sequential([
    layers.Rescaling(1./255),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomShear(0.1),
    ])
# Applies function to every image and label pair in pipeline
train_data = train_data.map(lambda x, y: (data_augmentation(x, training=True), y))

# Validation and test sets only rescaling 
rescale = layers.Rescaling(1./255)
val_data  = val_data.map(lambda x, y: (rescale(x), y))
test_data = test_data.map(lambda x, y: (rescale(x), y))

# Improve data pipeline to decrease computational time 
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.prefetch(AUTOTUNE)
val_data   = val_data.prefetch(AUTOTUNE)
test_data  = test_data.prefetch(AUTOTUNE)
 

# 2.0 Neural Network Architecture Design
mdl1 = keras.Sequential([
    layers.Conv2D(32,(3,3), activation='relu', input_shape=(500,500,3)),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64,(3,3), activation='relu', input_shape=(500,500,3)),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(128,(3,3), activation='relu', input_shape=(500,500,3)),
    layers.MaxPooling2D((2,2)),
        
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5), 
    layers.Dense(3, activation="softmax")
    ])


mdl1.summary() #visualize model structure and parameter numbers 

mdl1.compile (
    optimizer = keras.optimizers.Adam(learning_rate=1e-3),
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]
    )
#early stop to prevent overfitting 
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True
    )
#training model
EPOCHS = 15
history1 = mdl1.fit(
    train_data,
    validation_data = val_data,
    epochs = EPOCHS,
    callbacks=[early_stop],
    verbose = 1
    )
# # evaluate final model on test dataset 
# test_loss, test_acc = mdl1.evaluate(test_data, verbose=0)
# print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

# 3.0 Hyperparameter Analysis 
# defining function to change hyperparameters easily 
def build_model_flatten(conv_activation="relu",
                        dense_activation="relu",
                        filters=(32,64,128),
                        dense_units = 128,
                        dropout_rate=0.5):
# build the same CNN as before but allows changes to activations, filter sizes
# and dense layer width

    model = keras.Sequential([
        layers.Conv2D(filters[0], (3,3), activation=conv_activation, input_shape=(500,500,3)),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(filters[1], (3,3), activation=conv_activation),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(filters[2], (3,3), activation=conv_activation),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(dense_units, activation=dense_activation),
        layers.Dropout(dropout_rate),
        layers.Dense(3, activation="softmax")
    ])
    return model

# using Leaky ReLU instead of ReLU
def build_model_leaky(filters = (32,64,128),
                      dense_units = 128,
                      dense_activation="relu",
                      alpha=0.1,
                      dropout_rate=0.5):

    model = keras.Sequential([
        layers.Conv2D(filters[0], (3,3), input_shape=(500,500,3)),
        layers.LeakyReLU(alpha=alpha),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(filters[1], (3,3)),
        layers.LeakyReLU(alpha=alpha),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(filters[2], (3,3)),
        layers.LeakyReLU(alpha=alpha),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(dense_units, activation=dense_activation),
        layers.Dropout(dropout_rate),
        layers.Dense(3, activation="softmax")
    ])
    return model
# train and return the best model accuracy 
def train_and_best_val_acc(model, optimizer, label, epochs = 10):
    model.compile(optimizer=optimizer,
                  loss = "categorical_crossentropy",
                  metrics=["accuracy"])
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks = [early_stop],
        verbose = 1
    )
    # takes best validation accuracy across epochs 
    best_val_acc = float(max(history.history["val_accuracy"]))
    print(f"{label:20s} --> Best Val Accuracy = {best_val_acc:.4f}")
    return best_val_acc

results = {}


# mdl2 LeakyReLU activation
mdl2_leaky = build_model_leaky()
results["mdl2_LeakyReLU"] = train_and_best_val_acc(
    mdl2_leaky, keras.optimizers.Adam(1e-3),
    "mdl2: LeakyReLU convs (alpha=0.1)"
)

# mdl3 ELU in dense layer
mdl3_elu = build_model_flatten(dense_activation="elu")
results["mdl3_ELU_dense"] = train_and_best_val_acc(
    mdl3_elu, keras.optimizers.Adam(1e-3),
    "mdl3: Dense activation = ELU"
)


print("\n Best Validation Accuracy for mdl 2 and 3")
for name, acc in results.items():
    print(f"{name:10s}: {acc:.4f}")


# 4.0 Model Evaluation 
def plot_history(history, title_prefix=""):
    acc      = history.history.get("accuracy", [])
    val_acc  = history.history.get("val_accuracy", [])
    loss     = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs   = range(1, len(acc) + 1)
    
    # Accuracy
    plt.figure(figsize=(6,4))
    plt.plot(epochs, acc,     label="Training Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} Training vs Validation Accuracy")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    
    # Loss
    plt.figure(figsize=(6,4))
    plt.plot(epochs, loss,     label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Training vs Validation Loss")
    plt.legend(); plt.grid(True)
    plt.tight_layout()

plot_history(history1, title_prefix="Model 1:")

test_loss, test_acc = mdl1.evaluate(test_data, verbose=0)
print(f"[TEST] accuracy = {test_acc:.4f} | loss = {test_loss:.4f}")

mdl1.save("best_model.h5") #saving model and its trained weights to be called in next step
import json
with open("class_names.json","w") as f:
    json.dump(class_names, f)