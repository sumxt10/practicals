
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()


x_train = x_train / 255.0
x_test = x_test / 255.0


x_train = x_train[..., None]
x_test = x_test[..., None]


y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)


def build_model():
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


def train_model(name, loss_fn):
    print(f"\nTraining with {name}\n")

    model = build_model()

    model.compile(
        optimizer='adam',
        loss=loss_fn,
        metrics=['accuracy']
    )

    history = model.fit(
        x_train, y_train_cat,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    test_acc = model.evaluate(x_test, y_test_cat, verbose=0)[1]
    print(f"{name} Test Accuracy: {test_acc:.4f}")

    return history



hist_cce = train_model("Categorical Crossentropy", 'categorical_crossentropy')
hist_bce = train_model("Binary Crossentropy", 'binary_crossentropy')
hist_mse = train_model("Mean Squared Error", 'mse')




plt.figure()
plt.plot(hist_cce.history['val_accuracy'], label='CCE')
plt.plot(hist_bce.history['val_accuracy'], label='BCE')
plt.plot(hist_mse.history['val_accuracy'], label='MSE')
plt.title("Validation Accuracy Comparison (Fashion-MNIST)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


plt.figure()
plt.plot(hist_cce.history['val_loss'], label='CCE')
plt.plot(hist_bce.history['val_loss'], label='BCE')
plt.plot(hist_mse.history['val_loss'], label='MSE')
plt.title("Validation Loss Comparison (Fashion-MNIST)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()