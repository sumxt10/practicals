
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()


x_train = x_train / 255.0
x_test = x_test / 255.0


x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_test_cnn = x_test.reshape(-1, 28, 28, 1)


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)



def build_ann_baseline():
    return keras.Sequential([
        keras.Input(shape=(28,28)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

def build_ann_regularized():
    return keras.Sequential([
        keras.Input(shape=(28,28)),
        layers.Flatten(),
        layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(0.00005)),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(0.00005)),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(0.00005)),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])



def build_cnn_baseline():
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

def build_cnn_regularized():
    return keras.Sequential([
        keras.Input(shape=(28,28,1)),
        layers.Conv2D(32, (3,3), activation='relu',
                      kernel_regularizer=regularizers.l2(0.00005)),
        layers.Dropout(0.2),
        layers.Conv2D(64, (3,3), activation='relu',
                      kernel_regularizer=regularizers.l2(0.00005)),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Conv2D(128, (3,3), activation='relu',
                      kernel_regularizer=regularizers.l2(0.00005)),
        layers.Flatten(),
        layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(0.00005)),
        layers.Dropout(0.4),
        layers.Dense(10, activation='softmax')
    ])


def train_model(model, x_tr, y_tr, x_te, y_te, name, use_es=False):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    print(f"\nTraining {name}\n")
    
    history = model.fit(
        x_tr, y_tr,
        epochs=25,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop] if use_es else [],
        verbose=1
    )
    
    test_loss, test_acc = model.evaluate(x_te, y_te, verbose=0)
    print(f"{name} Test Accuracy: {test_acc:.4f}")
    
    return history



ann_base_hist = train_model(build_ann_baseline(), x_train, y_train, x_test, y_test,
                           "ANN Baseline (Overfitting Expected)", False)

ann_reg_hist = train_model(build_ann_regularized(), x_train, y_train, x_test, y_test,
                          "ANN Regularized (Dropout + L2 + ES)", True)

cnn_base_hist = train_model(build_cnn_baseline(), x_train_cnn, y_train, x_test_cnn, y_test,
                           "CNN Baseline (Overfitting Expected)", False)

cnn_reg_hist = train_model(build_cnn_regularized(), x_train_cnn, y_train, x_test_cnn, y_test,
                          "CNN Regularized (Dropout + L2 + ES)", True)




plt.figure()

plt.plot(ann_base_hist.history['val_accuracy'], label='ANN Baseline')
plt.plot(ann_reg_hist.history['val_accuracy'], label='ANN Regularized')
plt.plot(cnn_base_hist.history['val_accuracy'], label='CNN Baseline')
plt.plot(cnn_reg_hist.history['val_accuracy'], label='CNN Regularized')

plt.title("Validation Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



plt.figure()

plt.plot(ann_base_hist.history['val_loss'], label='ANN Baseline')
plt.plot(ann_reg_hist.history['val_loss'], label='ANN Regularized')
plt.plot(cnn_base_hist.history['val_loss'], label='CNN Baseline')
plt.plot(cnn_reg_hist.history['val_loss'], label='CNN Regularized')

plt.title("Validation Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()