
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()


x_train = x_train / 255.0
x_test = x_test / 255.0


x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)



def build_baseline():
    return keras.Sequential([
        keras.Input(shape=(28,28,1)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

baseline = build_baseline()

baseline.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining BASELINE (Overfitting expected)\n")

hist_base = baseline.fit(
    x_train, y_train,
    epochs=25,                
    batch_size=64,
    validation_split=0.2,
    verbose=1
)



def build_improved():
    return keras.Sequential([
        keras.Input(shape=(28,28,1)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

improved = build_improved()

improved.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    verbose=1
)

print("\nTraining IMPROVED (LR Scheduling + EarlyStopping)\n")

hist_imp = improved.fit(
    x_train, y_train,
    epochs=25,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)



base_acc = baseline.evaluate(x_test, y_test, verbose=0)[1]
imp_acc = improved.evaluate(x_test, y_test, verbose=0)[1]

print(f"\nBaseline Test Accuracy: {base_acc:.4f}")
print(f"Improved Test Accuracy: {imp_acc:.4f}")




plt.figure()
plt.plot(hist_base.history['val_accuracy'], label='Baseline')
plt.plot(hist_imp.history['val_accuracy'], label='Improved')
plt.title("Validation Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


plt.figure()
plt.plot(hist_base.history['val_loss'], label='Baseline')
plt.plot(hist_imp.history['val_loss'], label='Improved')
plt.title("Validation Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()