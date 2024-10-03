import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import load_model

# Load the model from the specified file
loaded_model = load_model('app/models/regression.h5')

if __name__ == '__main__':
    # Load and preprocess the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values to [0, 1]

    # Compile the model with optimizer, loss, and metrics
    loaded_model.compile(optimizer='sgd',
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                         metrics=['sparse_categorical_accuracy'])

    # Define checkpoint path for saving the model
    checkpoint_save_path = 'checkpoint/Regression.ckpt'
    if os.path.exists(checkpoint_save_path + '.index'):
        loaded_model.load_weights(checkpoint_save_path)  # Load weights if checkpoint exists

    # Callback to save the best weights during training
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True)

    # Train the model
    history = loaded_model.fit(x_train, y_train,
                               batch_size=32,
                               epochs=20,
                               validation_data=(x_test, y_test),
                               validation_freq=1,
                               callbacks=[cp_callback])

    # Save the trained model to a file
    loaded_model.save('app/models/regression.h5')

    # Display a summary of the model architecture
    loaded_model.summary()

    # Retrieve history for accuracy and loss
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()
