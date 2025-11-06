#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: nekaumakanth
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tensorflow import keras
from tensorflow.keras.preprocessing import image

IMG_SIZE = (500, 500)

# load trained model
model = keras.models.load_model("best_model.h5")

# load class names from file
with open("class_names.json", "r") as f:  
    class_names = json.load(f)

print("Loaded class names:", class_names)


def load_and_preprocess(img_path):
    pil_img = image.load_img(img_path, target_size=IMG_SIZE) # loading image
    arr = image.img_to_array(pil_img).astype("float32") / 255.0 # array and normalizing
    arr = np.expand_dims(arr, axis=0)     
    return arr, pil_img


def predict_and_display(img_path):
    # getting true label 
    true_label = os.path.basename(os.path.dirname(img_path))
    
    x, pil_img = load_and_preprocess(img_path)
    probs = model.predict(x, verbose=0)[0] #outputs probabilities 
    pred_index = int(np.argmax(probs)) # gets index of highest probability 
    pred_label = class_names[pred_index]
    pred_prob  = float(probs[pred_index])

    print(f"\nImage: {img_path}")
    print(f"True Label:      {true_label}")
    print(f"Predicted Label: {pred_label} (Probability: {pred_prob:.3f})\n")

    print("Class Probabilities:")
    for name, p in zip(class_names, probs):
        print(f"  {name:>12s}: {p:.3f}")

    # displaying images 
    plt.figure(figsize=(5,5))
    plt.imshow(pil_img)
    plt.axis("off")

 
    plt.title(f"Predicted: {pred_label} ({pred_prob:.1%})\nTrue: {true_label}", fontsize=12)

    # probability on the image
    text = "\n".join([f"{name}: {p:.2f}" for name, p in zip(class_names, probs)])
    plt.text(
    0.98, 0.02, text,                      
    transform=plt.gca().transAxes,        
    ha="right", va="bottom",          
    color="white", fontsize=10,
    bbox=dict(facecolor="black", alpha=0.6)
)


test_images = [
    "Data/test/crack/test_crack.jpg",
    "Data/test/missing-head/test_missinghead.jpg",
    "Data/test/paint-off/test_paintoff.jpg",
]

for img_path in test_images:
    predict_and_display(img_path)
