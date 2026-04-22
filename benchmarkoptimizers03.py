import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ── Dataset: Fashion-MNIST ────────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

x_train = x_train / 255.0
x_test  = x_test  / 255.0

x_train = x_train.reshape(-1, 28*28)
x_test  = x_test.reshape(-1, 28*28)


def build_model():
    return keras.Sequential([
        keras.Input(shape=(784,)),

        # Deeper network — optimizer differences show more clearly
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64,  activation='relu'),

        layers.Dense(10, activation='softmax')
    ])


optimizers = {
    "SGD":      keras.optimizers.SGD(learning_rate=0.01),       # baseline, slowest
    "Adam":     keras.optimizers.Adam(learning_rate=0.001),     # adaptive, fast
    "RMSprop":  keras.optimizers.RMSprop(learning_rate=0.001),  # adaptive, similar to Adam
}

histories = {}
results   = {}

for name, opt in optimizers.items():
    print(f"\n{'='*50}")
    print(f"Training with {name}")
    print(f"{'='*50}")

    model = build_model()
    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        x_train, y_train,
        epochs=20,           # more epochs — SGD needs time to catch up
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    histories[name] = history
    results[name]   = test_acc

# ── Final results ─────────────────────────────────────────────────────────
print("\n" + "="*50)
print("Final Test Accuracies:")
print("="*50)
for name in results:
    print(f"  {name:<10} → {results[name]:.4f}")

# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, metric, title in zip(
    axes,
    ['val_accuracy', 'val_loss'],
    ['Validation Accuracy (Fashion-MNIST)', 'Validation Loss (Fashion-MNIST)']
):
    for name in histories:
        ax.plot(histories[name].history[metric], label=name, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(metric.split('_')[1].capitalize())
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()