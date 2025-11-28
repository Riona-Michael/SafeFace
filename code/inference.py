# code/inference.py
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt

MODEL_PATH = "./results/best_model.h5"
IMG_PATH = "./sample_test.jpg"   # place test image in project root
IMG_SIZE = (128, 128)

def load_img_for_model(img_path):
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr, img

def predict(img_path):
    model = load_model(MODEL_PATH)
    arr, pil_img = load_img_for_model(img_path)
    prob = model.predict(arr)[0][0]
    label = "fake" if prob >= 0.5 else "real"
    return label, float(prob), pil_img

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found at:", MODEL_PATH)
        exit()

    if not os.path.exists(IMG_PATH):
        print("❌ Test image missing. Put an image at:", IMG_PATH)
        exit()

    label, prob, img = predict(IMG_PATH)

    print(f"\nPrediction → {label.upper()}  (prob={prob:.4f})\n")

    # Save result image
    os.makedirs("./results", exist_ok=True)
    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.title(f"{label} ({prob:.2f})")
    plt.axis('off')
    out_path = "./results/sample_prediction.png"
    plt.savefig(out_path, bbox_inches='tight')

    print("Saved prediction figure at →", out_path)
