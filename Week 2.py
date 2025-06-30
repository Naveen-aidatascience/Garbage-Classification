import os
import shutil
import random
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

print("TensorFlow Version:", tf.__version__)
def convert_and_resize_images(folder, size=(224, 224)):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(root, file)
                img = Image.open(path).convert("RGB")
                img = img.resize(size)
                webp_path = os.path.splitext(path)[0] + ".webp"
                img.save(webp_path, "webp", quality=80)
                os.remove(path)  # Remove original image

convert_and_resize_images("Garbage Classification dataset")
def split_dataset(source_dir, train_dir, test_dir, split_ratio=0.8):
    classes = os.listdir(source_dir)
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        if not os.path.isdir(cls_path): continue

        images = [img for img in os.listdir(cls_path) if img.endswith(".webp")]
        random.shuffle(images)
        split_idx = int(len(images) * split_ratio)

        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

        for i, img in enumerate(images):
            src = os.path.join(cls_path, img)
            dest = os.path.join(train_dir if i < split_idx else test_dir, cls, img)
            shutil.copyfile(src, dest)

split_dataset(
    "Garbage Classification dataset",
    "dataset/train",
    "dataset/test"
)
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    "dataset/train", target_size=(224, 224), batch_size=32, class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    "dataset/test", target_size=(224, 224), batch_size=32, class_mode='categorical'
)
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10
)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

