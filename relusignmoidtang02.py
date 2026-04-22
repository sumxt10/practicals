# Import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# Function to build model with different activation
def build_model(activation):
    model = keras.Sequential([
        keras.Input(shape=(784,)),
        layers.Dense(128, activation=activation),
        layers.Dense(64, activation=activation),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Train models with different activations
activations = ['relu', 'sigmoid', 'tanh']
histories = {}
results = {}

for act in activations:
    print(f"\nTraining with {act.upper()} activation\n")
    model = build_model(act)
    
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    histories[act] = history
    results[act] = test_acc

# Print results
print("\nFinal Test Accuracies:")
for act in results:
    print(f"{act.upper()}: {results[act]:.4f}")

# Plot Accuracy Comparison
plt.figure()
for act in activations:
    plt.plot(histories[act].history['val_accuracy'], label=act.upper())

plt.title("Validation Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Plot Loss Comparison
plt.figure()
for act in activations:
    plt.plot(histories[act].history['val_loss'], label=act.upper())

plt.title("Validation Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()