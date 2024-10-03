import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Load and preprocess the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Reshape to include the channel dimension (1 for grayscale images)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # Normalize pixel values to [0, 1]
    x_train, x_test = x_train / 255.0, x_test / 255.0  

    # Define the CNN model architecture
    model = tf.keras.models.Sequential([
        Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=36, kernel_size=(5, 5), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')  # Output layer for 10 classes
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])

    # Define checkpoint path for saving the model weights
    checkpoint_save_path = 'checkpoint/convolutional.ckpt'

    # Load weights if checkpoint exists
    if os.path.exists(checkpoint_save_path + '.index'):
        model.load_weights(checkpoint_save_path)

    # Define callback to save the best weights during training
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True)

    # Train the model
    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=20,
                        validation_data=(x_test, y_test),
                        validation_freq=1,
                        callbacks=[cp_callback])

    # Save the model after training
    model.save('convolutional.h5')

    # Display the model architecture summary
    model.summary()

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
