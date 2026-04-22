
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


vocab_size = 10000
maxlen = 200

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)


x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)



def build_baseline():
    return keras.Sequential([
        layers.Embedding(vocab_size, 128),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])



def build_batchnorm():
    return keras.Sequential([
        layers.Embedding(vocab_size, 128),
        layers.Flatten(),

        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.Dense(1, activation='sigmoid')
    ])


def compile_model(model):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )


def train_model(model, name):
    print(f"\nTraining {name}\n")
    history = model.fit(
        x_train, y_train,
        epochs=8,              
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )
    return history



baseline = build_baseline()
compile_model(baseline)
hist_base = train_model(baseline, "Baseline MLP")

bn_model = build_batchnorm()
compile_model(bn_model)
hist_bn = train_model(bn_model, "MLP with BatchNorm")



base_acc = baseline.evaluate(x_test, y_test, verbose=0)[1]
bn_acc = bn_model.evaluate(x_test, y_test, verbose=0)[1]

print(f"\nBaseline Accuracy: {base_acc:.4f}")
print(f"BatchNorm Accuracy: {bn_acc:.4f}")




plt.figure()
plt.plot(hist_base.history['val_accuracy'], label='Baseline')
plt.plot(hist_bn.history['val_accuracy'], label='BatchNorm')
plt.title("Validation Accuracy (Convergence Comparison)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


plt.figure()
plt.plot(hist_base.history['val_loss'], label='Baseline')
plt.plot(hist_bn.history['val_loss'], label='BatchNorm')
plt.title("Validation Loss (Stability Comparison)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()