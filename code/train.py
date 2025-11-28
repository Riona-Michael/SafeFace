# code/train.py
import os, random, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers, callbacks

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = "./dataset/processed"
IMG_SIZE = (128, 128)
BATCH_SIZE = 4          # smaller batch because dataset is tiny
EPOCHS = 5
MODEL_DIR = "./results"
os.makedirs(MODEL_DIR, exist_ok=True)

# ❌ NO validation_split here
datagen = ImageDataGenerator(rescale=1./255,
                             horizontal_flip=True,
                             rotation_range=10)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    seed=SEED
)

# ❌ No validation generator (your dataset is too small)
val_gen = None

# Load MobileNetV2
base = MobileNetV2(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
                   include_top=False, weights='imagenet')
base.trainable = False

# Build Model
model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=optimizers.Adam(1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Only save best from training data (no validation)
ckpt_path = os.path.join(MODEL_DIR, "best_model.h5")
cb = [callbacks.ModelCheckpoint(ckpt_path, monitor='accuracy', save_best_only=True),
      callbacks.EarlyStopping(monitor='accuracy', patience=3, restore_best_weights=True)]

# Train (no validation_data)
history = model.fit(train_gen, epochs=EPOCHS, callbacks=cb)

# Save final model
model.save(os.path.join(MODEL_DIR, "final_model.h5"))

# Save training metrics
metrics = {k: [float(x) for x in v] for k, v in history.history.items()}
with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f)

print("Training completed. Models saved in", MODEL_DIR)
