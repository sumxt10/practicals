import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train / 255.0
x_test  = x_test  / 255.0
y_train = y_train.flatten()
y_test  = y_test.flatten()

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat  = keras.utils.to_categorical(y_test,  10)


def build_model(output_activation='softmax'):
    return keras.Sequential([
        keras.Input(shape=(32, 32, 3)),

        
        layers.Conv2D(32, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        
        layers.Conv2D(64, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        
        layers.Conv2D(128, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(10, activation=output_activation)  
    ])


def train_model(loss_name, loss_fn, output_activation, y_tr, y_te, epochs=20):
    print(f"\n{'='*50}\nTraining with {loss_name}\n{'='*50}")

    model = build_model(output_activation=output_activation)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=loss_fn,
        metrics=['accuracy']
    )

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, verbose=1),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=7, restore_best_weights=True)
    ]

    history = model.fit(
        x_train, y_tr,
        epochs=epochs,
        batch_size=128,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    test_acc = model.evaluate(x_test, y_te, verbose=0)[1]
    print(f"\n{loss_name} → Test Accuracy: {test_acc:.4f}")
    return history



hist_cce = train_model(
    "Categorical Crossentropy",
    'categorical_crossentropy',
    output_activation='softmax',   
    y_tr=y_train_cat,
    y_te=y_test_cat
)




hist_bce = train_model(
    "Binary Crossentropy",
    'binary_crossentropy',
    output_activation='sigmoid',   
    y_tr=y_train_cat,
    y_te=y_test_cat
)


hist_mse = train_model(
    "Mean Squared Error",
    'mse',
    output_activation='softmax',
    y_tr=y_train_cat,
    y_te=y_test_cat
)



fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, metric, title in zip(
    axes,
    ['val_accuracy', 'val_loss'],
    ['Validation Accuracy (CIFAR-10)', 'Validation Loss (CIFAR-10)']
):
    ax.plot(hist_cce.history[metric], label='CCE', linewidth=2)
    ax.plot(hist_bce.history[metric], label='BCE', linewidth=2)
    ax.plot(hist_mse.history[metric], label='MSE', linewidth=2)
    ax.set_title(title)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(metric.split('_')[1].capitalize())
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


for name, hist in [("CCE", hist_cce), ("BCE", hist_bce), ("MSE", hist_mse)]:
    best_val = max(hist.history['val_accuracy'])
    print(f"{name}: best val_accuracy = {best_val:.4f}")