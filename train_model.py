import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# =========================================================
# SETTINGS
# =========================================================

DATASET_DIR = "dataset"
MODEL_NAME = "waste_model.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123

INITIAL_EPOCHS = 25
FINE_TUNE_EPOCHS = 15

CLASS_NAMES = [
    "cardboard",
    "e_waste",
    "glass",
    "metal",
    "other",
    "paper",
    "plastic",
    "trash"
]

AUTOTUNE = tf.data.AUTOTUNE

# =========================================================
# SEED
# =========================================================

tf.random.set_seed(SEED)
np.random.seed(SEED)

print("=" * 65)
print("AI WASTE CLASSIFICATION")
print("BALANCED TRAIN / VALIDATION SPLIT")
print("=" * 65)

# =========================================================
# COLLECT ALL IMAGE FILES
# =========================================================

all_files = []
all_labels = []

print("\nDATASET COUNTS")
print("-" * 65)

for class_index, class_name in enumerate(CLASS_NAMES):

    folder = os.path.join(
        DATASET_DIR,
        class_name
    )

    if not os.path.exists(folder):

        raise FileNotFoundError(
            f"Folder not found: {folder}"
        )

    files = []

    for filename in os.listdir(folder):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):

            files.append(
                os.path.join(
                    folder,
                    filename
                )
            )

    print(
        f"{class_name:10s}: {len(files)} images"
    )

    all_files.extend(files)

    all_labels.extend(
        [class_index] * len(files)
    )

all_files = np.array(all_files)
all_labels = np.array(all_labels)

print("-" * 65)
print(
    "TOTAL IMAGES:",
    len(all_files)
)

# =========================================================
# BALANCED STRATIFIED SPLIT
# =========================================================

print("\nCreating balanced 80/20 split...")

train_files, val_files, train_labels, val_labels = train_test_split(

    all_files,

    all_labels,

    test_size=0.20,

    random_state=SEED,

    stratify=all_labels
)

print(
    "Training images:",
    len(train_files)
)

print(
    "Validation images:",
    len(val_files)
)

# =========================================================
# CHECK VALIDATION CLASS COUNTS
# =========================================================

print("\nVALIDATION CLASS COUNTS")
print("-" * 65)

for index, class_name in enumerate(CLASS_NAMES):

    count = np.sum(
        val_labels == index
    )

    print(
        f"{class_name:10s}: {count}"
    )

# =========================================================
# IMAGE LOADING FUNCTION
# =========================================================

def load_image(
    filepath,
    label
):

    image = tf.io.read_file(
        filepath
    )

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = preprocess_input(
        image
    )

    return image, label

# =========================================================
# CREATE TF DATASET
# =========================================================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        train_files,
        train_labels
    )
)

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (
        val_files,
        val_labels
    )
)

# =========================================================
# SHUFFLE TRAINING
# =========================================================

train_dataset = train_dataset.shuffle(
    buffer_size=len(train_files),
    seed=SEED
)

# =========================================================
# LOAD IMAGES
# =========================================================

train_dataset = train_dataset.map(
    load_image,
    num_parallel_calls=AUTOTUNE
)

validation_dataset = validation_dataset.map(
    load_image,
    num_parallel_calls=AUTOTUNE
)

# =========================================================
# BATCH
# =========================================================

train_dataset = train_dataset.batch(
    BATCH_SIZE
)

validation_dataset = validation_dataset.batch(
    BATCH_SIZE
)

# =========================================================
# PREFETCH
# =========================================================

train_dataset = train_dataset.prefetch(
    AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    AUTOTUNE
)

# =========================================================
# DATA AUGMENTATION
# =========================================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.10
    ),

    layers.RandomZoom(
        0.15
    ),

    layers.RandomContrast(
        0.10
    )

])

# =========================================================
# MOBILE NET V2
# =========================================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(

    weights="imagenet",

    include_top=False,

    input_shape=(
        224,
        224,
        3
    )
)

base_model.trainable = False

# =========================================================
# BUILD MODEL
# =========================================================

inputs = layers.Input(
    shape=(
        224,
        224,
        3
    )
)

x = data_augmentation(
    inputs
)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(
    0.35
)(x)

x = layers.Dense(
    256,
    activation="relu"
)(x)

x = layers.Dropout(
    0.25
)(x)

outputs = layers.Dense(
    len(CLASS_NAMES),
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs
)

# =========================================================
# COMPILE
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)

# =========================================================
# CALLBACKS
# =========================================================

callbacks = [

    ModelCheckpoint(
        MODEL_NAME,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

# =========================================================
# PHASE 1
# =========================================================

print("\n")
print("=" * 65)
print("PHASE 1 - TRANSFER LEARNING")
print("=" * 65)

model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=INITIAL_EPOCHS,

    callbacks=callbacks
)

# =========================================================
# LOAD BEST MODEL
# =========================================================

model = tf.keras.models.load_model(
    MODEL_NAME,
    compile=False
)

# =========================================================
# FINE TUNING
# =========================================================

print("\n")
print("=" * 65)
print("PHASE 2 - FINE TUNING")
print("=" * 65)

base_model = None

for layer in model.layers:

    if isinstance(
        layer,
        tf.keras.Model
    ):

        base_model = layer
        break

if base_model is not None:

    base_model.trainable = True

    for layer in base_model.layers[:-30]:

        layer.trainable = False

    for layer in base_model.layers:

        if isinstance(
            layer,
            layers.BatchNormalization
        ):

            layer.trainable = False

# =========================================================
# RECOMPILE
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)

# =========================================================
# FINE TUNE
# =========================================================

model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=FINE_TUNE_EPOCHS,

    callbacks=callbacks
)

# =========================================================
# LOAD BEST MODEL
# =========================================================

model = tf.keras.models.load_model(
    MODEL_NAME,
    compile=False
)

# =========================================================
# FINAL EVALUATION
# =========================================================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)

print("\n")
print("=" * 65)
print("FINAL VALIDATION")
print("=" * 65)

loss, accuracy = model.evaluate(
    validation_dataset,
    verbose=1
)

print()
print(
    f"FINAL VALIDATION ACCURACY: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"FINAL VALIDATION LOSS: "
    f"{loss:.4f}"
)

# =========================================================
# CLASS-WISE ACCURACY
# =========================================================

print("\n")
print("=" * 65)
print("CLASS-WISE ACCURACY")
print("=" * 65)

correct = {
    class_name: 0
    for class_name in CLASS_NAMES
}

total = {
    class_name: 0
    for class_name in CLASS_NAMES
}

# =========================================================
# PREDICT VALIDATION DATA
# =========================================================

for images, labels in validation_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    true_labels = labels.numpy()

    for true_label, predicted_label in zip(
        true_labels,
        predicted_labels
    ):

        true_label = int(
            true_label
        )

        predicted_label = int(
            predicted_label
        )

        class_name = CLASS_NAMES[
            true_label
        ]

        total[class_name] += 1

        if true_label == predicted_label:

            correct[class_name] += 1

# =========================================================
# PRINT CLASS ACCURACY
# =========================================================

for class_name in CLASS_NAMES:

    if total[class_name] > 0:

        class_accuracy = (
            correct[class_name]
            /
            total[class_name]
        ) * 100

    else:

        class_accuracy = 0

    print(
        f"{class_name:10s}: "
        f"{correct[class_name]}/"
        f"{total[class_name]} "
        f"= {class_accuracy:.2f}%"
    )

# =========================================================
# SAVE FINAL MODEL
# =========================================================

model.save(
    MODEL_NAME
)

print("\n")
print("=" * 65)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 65)

print(
    "Model:",
    MODEL_NAME
)

print(
    f"Final Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    "\nTraining completed."
)