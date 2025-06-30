import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# --- Configuration ---
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-5  # Lower learning rate for fine-tuning
DROPOUT_RATE = 0.5
L2_REGULARIZER = 0.01
MANUAL_PRESENT_CLASS_WEIGHT_MULTIPLIER = 1.5  # Adjust class weight multiplier for imbalance
PREDICTION_THRESHOLD = 0.7

# Paths to dataset
train_dir = "/home/akshita-bindal/Desktop/new_manual/dataset_split/train"
val_dir = "/home/akshita-bindal/Desktop/new_manual/dataset_split/val"
test_dir = "/home/akshita-bindal/Desktop/new_manual/dataset_split/test"
output_dir = "./model_output"
os.makedirs(output_dir, exist_ok=True)

# --- Data Generators ---
print("Setting up data generators...")
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.25,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)
val_test_datagen = ImageDataGenerator(rescale=1.0/255)

train_data = train_datagen.flow_from_directory(
    train_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='binary'
)
val_data = val_test_datagen.flow_from_directory(
    val_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='binary'
)
test_data = val_test_datagen.flow_from_directory(
    test_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='binary', shuffle=False
)

# --- Build the Model ---
print("\nBuilding the model...")
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_model.trainable = False  # Freeze base model for initial training

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZER)),
    Dropout(DROPOUT_RATE),
    Dense(1, activation="sigmoid")
])

# Compile with a low learning rate for fine-tuning
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# --- Fine-tune Layers ---
fine_tune_at = len(base_model.layers) - 30
if fine_tune_at < 0:
    fine_tune_at = 0
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True

# --- Class Weights ---
classes = np.unique(train_data.classes)
class_weights_array = compute_class_weight(class_weight='balanced', classes=classes, y=train_data.classes)
class_weights_dict = {
    0: class_weights_array[0],  # Manual Absent
    1: class_weights_array[1] * MANUAL_PRESENT_CLASS_WEIGHT_MULTIPLIER  # Manual Present
}
print(f"Class weights: {class_weights_dict}")

# --- Callbacks ---
early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1)

# --- Train the Model ---
print("\nTraining the model...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights_dict,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Save the trained model
model_save_path = os.path.join(output_dir, "manual_classifier_mobilenetv2.h5")
model.save(model_save_path, include_optimizer=True)
print(f"\nModel saved to: {model_save_path}")

# --- Evaluate the Model ---
print("\nEvaluating the model on test data...")
loss, accuracy = model.evaluate(test_data, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# --- Confusion Matrix and Classification Report ---
print("\nGenerating classification report...")
test_labels = test_data.classes
predictions_raw = model.predict(test_data, verbose=1)
predicted_labels = (predictions_raw > PREDICTION_THRESHOLD).astype(int).flatten()

conf_matrix = confusion_matrix(test_labels, predicted_labels)
class_report = classification_report(test_labels, predicted_labels, target_names=['Manual Absent', 'Manual Present'])

print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Save classification report
report_path = os.path.join(output_dir, "classification_report.txt")
with open(report_path, "w") as f:
    f.write("--- Confusion Matrix ---\n")
    f.write(str(conf_matrix))
    f.write("\n\n--- Classification Report ---\n")
    f.write(class_report)
print(f"Classification report saved to: {report_path}")

# --- Identify and Print Misclassified Image Names ---
print("\nIdentifying misclassified images...")

# Get the class indices mapping from the generator
# class_indices: {'Manual Absent': 0, 'Manual Present': 1}
class_indices = test_data.class_indices
inverse_class_indices = {v: k for k, v in class_indices.items()} # {0: 'Manual Absent', 1: 'Manual Present'}

# --- Plot Training History ---
print("\nPlotting training history...")
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'))

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
print(f"Accuracy and Loss plots saved to: {output_dir}")
