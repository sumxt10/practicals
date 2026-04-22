import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

x_train = x_train / 255.0
x_test  = x_test  / 255.0
x_train = x_train[..., None]
x_test  = x_test[..., None]

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat  = keras.utils.to_categorical(y_test,  10)


def build_model(output_activation='softmax'):
    """output_activation: 'softmax' for CCE/MSE, 'sigmoid' for BCE"""
    return keras.Sequential([
        keras.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(10, activation=output_activation)  
    ])


def train_model(name, loss_fn, output_activation='softmax', epochs=15):
    print(f"\n{'='*50}")
    print(f"Training with {name}")
    print(f"{'='*50}")

    model = build_model(output_activation=output_activation)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss_fn,
        metrics=['accuracy']
    )

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, verbose=1
    )

    history = model.fit(
        x_train, y_train_cat,   
        epochs=epochs,
        batch_size=64,
        validation_split=0.2,
        callbacks=[lr_scheduler],
        verbose=1
    )

    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"\n{name} → Test Accuracy: {test_acc:.4f}")
    return history



hist_cce = train_model("Categorical Crossentropy", 
                        'categorical_crossentropy', 
                        output_activation='softmax')



hist_bce = train_model("Binary Crossentropy",       
                        'binary_crossentropy',       
                        output_activation='sigmoid') 


hist_mse = train_model("Mean Squared Error",        
                        'mse',                       
                        output_activation='softmax')



fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(hist_cce.history['val_accuracy'], label='CCE', linewidth=2)
axes[0].plot(hist_bce.history['val_accuracy'], label='BCE', linewidth=2)
axes[0].plot(hist_mse.history['val_accuracy'], label='MSE', linewidth=2)
axes[0].set_title("Validation Accuracy — Fashion MNIST")
axes[0].set_xlabel("Epochs"); axes[0].set_ylabel("Accuracy")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(hist_cce.history['val_loss'], label='CCE', linewidth=2)
axes[1].plot(hist_bce.history['val_loss'], label='BCE', linewidth=2)
axes[1].plot(hist_mse.history['val_loss'], label='MSE', linewidth=2)
axes[1].set_title("Validation Loss — Fashion MNIST")
axes[1].set_xlabel("Epochs"); axes[1].set_ylabel("Loss")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()