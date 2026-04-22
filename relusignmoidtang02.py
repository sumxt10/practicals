import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

x_train = x_train / 255.0
x_test  = x_test  / 255.0

x_train = x_train.reshape(-1, 28*28)
x_test  = x_test.reshape(-1, 28*28)


def build_model(activation):
    model = keras.Sequential([
        keras.Input(shape=(784,)),

        
        
        layers.Dense(256, activation=activation),
        layers.Dense(128, activation=activation),
        layers.Dense(64,  activation=activation),
        layers.Dense(32,  activation=activation),  

        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


activations = ['relu', 'sigmoid', 'tanh']
histories   = {}
results     = {}

for act in activations:
    print(f"\n{'='*50}")
    print(f"Training with {act.upper()} activation")
    print(f"{'='*50}")

    model = build_model(act)

    history = model.fit(
        x_train, y_train,
        epochs=20,              
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    histories[act] = history
    results[act]   = test_acc


print("\n" + "="*50)
print("Final Test Accuracies:")
print("="*50)
for act in activations:
    print(f"  {act.upper():<10} → {results[act]:.4f}")


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, metric, title in zip(
    axes,
    ['val_accuracy', 'val_loss'],
    ['Validation Accuracy (Fashion-MNIST)', 'Validation Loss (Fashion-MNIST)']
):
    for act in activations:
        ax.plot(histories[act].history[metric], label=act.upper(), linewidth=2)
    ax.set_title(title)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(metric.split('_')[1].capitalize())
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()