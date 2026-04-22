
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


x_train = x_train / 255.0
x_test = x_test / 255.0


x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)


def build_model():
    model = keras.Sequential([
        keras.Input(shape=(784,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model


optimizers = {
    "SGD": keras.optimizers.SGD(learning_rate=0.01),
    "Adam": keras.optimizers.Adam(),
    "RMSprop": keras.optimizers.RMSprop()
}

histories = {}
results = {}


for name, opt in optimizers.items():
    print(f"\nTraining with {name}\n")
    
    model = build_model()
    
    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    histories[name] = history
    results[name] = test_acc


print("\nFinal Test Accuracies:")
for name in results:
    print(f"{name}: {results[name]:.4f}")


plt.figure()
for name in histories:
    plt.plot(histories[name].history['val_accuracy'], label=name)

plt.title("Optimizer Comparison - Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


plt.figure()
for name in histories:
    plt.plot(histories[name].history['val_loss'], label=name)

plt.title("Optimizer Comparison - Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()