import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

IMG_SIZE = (224,224)
DEFAULT_MODEL_DIR = 'saved_models'

def evaluate_model(model, val_gen, save_dir="saved_models"):
    # Evaluate using Keras
    loss, acc = model.evaluate(val_gen, verbose=0)
    print("\n=== Evaluation on Validation Set ===")
    print(f"Validation Accuracy (Keras): {acc:.4f}")
    print(f"Validation Loss (Keras): {loss:.4f}")

    # Predictions
    y_true = val_gen.classes
    y_pred_probs = model.predict(val_gen, verbose=0).ravel()
    y_pred = (y_pred_probs > 0.5).astype(int)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=list(val_gen.class_indices.keys())))

    # Confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # ROC curve & AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0,1], [0,1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")

    os.makedirs(save_dir, exist_ok=True)
    roc_path = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()

    print(f"ROC curve saved to: {roc_path}")


def build_model(input_shape=(224,224,3)):
    base = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    base.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def get_generators(data_dir, batch_size=16):
    train_dir = os.path.join(data_dir,'augmented_train')
    val_dir = os.path.join(data_dir,'splits','val')

    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(train_dir, target_size=IMG_SIZE,
                                                  batch_size=batch_size, class_mode='binary', shuffle=True)
    val_gen = val_datagen.flow_from_directory(val_dir, target_size=IMG_SIZE,
                                              batch_size=batch_size, class_mode='binary', shuffle=False)
    return train_gen, val_gen

def plot_history(history, save_path):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc)+1)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, acc, 'bo-', label='Training Acc')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Acc')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train(data_dir, epochs=20, batch_size=16, model_dir=DEFAULT_MODEL_DIR):
    train_gen, val_gen = get_generators(data_dir, batch_size=batch_size)
    model = build_model()
    os.makedirs(model_dir, exist_ok=True)
    timestr = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(model_dir, f'brain_tumor_mobilenetv2_{timestr}.h5')

    callbacks = [
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        EarlyStopping(monitor='val_loss', patience=8, verbose=1, restore_best_weights=True)
    ]

    history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

    latest_model_file = os.path.join(model_dir,'brain_tumor_mobilenetv2_latest.h5')
    model.save(latest_model_file)

    # Save accuracy/loss plot
    plot_path = os.path.join(model_dir, f'training_history_{timestr}.png')
    plot_history(history, plot_path)

    # Print final metrics only
    best_val_acc = max(history.history['val_accuracy'])
    final_train_acc = history.history['accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]

    print("\n=== Training Summary ===")
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    print(f"Best model saved to: {model_path}")
    print(f"Latest model saved to: {latest_model_file}")
    print(f"Training plot saved to: {plot_path}")

    return model, val_gen, model_dir, history


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='dataset_prepared')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    model, val_gen, model_dir, history = train(args.data_dir, epochs=args.epochs, batch_size=args.batch_size)
    evaluate_model(model, val_gen, save_dir=model_dir)