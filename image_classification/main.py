import tensorflow
from tensorflow import keras
# from tensorflow.keras.models import load_model
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


dataset = keras.datasets.fashion_mnist  # import the dataset
((training_images, training_labels),(test_images, test_labels)) = dataset.load_data()  # defines the train and test dataset

'''
print(training_images.shape)
print(training_labels.shape)
print(test_images.shape)
print(test_labels.shape)

classifier_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']  # Class Labels

# Mount the subplot
for image in range(10):
  plt.subplot(2, 5, image+1)
  plt.imshow(training_images[image])
  plt.title(classifier_names[training_labels[image]])

plt.tight_layout()  # Ensures the layout is appropriate
plt.show()  # Show the images
'''

training_images = training_images/float(255)  # Normalization

# Defines the neural network parameters

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),  # Input layer
    keras.layers.Dense(256, activation=tensorflow.nn.relu),  # Hiden layer
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=tensorflow.nn.softmax)  # Output layer
])

# adam = keras.optimizers.Adam(lr=0.002)  # defines learning rate

stopping_criterion = [keras.callbacks.EarlyStopping(monitor='val_loss'),  # Stop the training if the validation loss get worse
                      keras.callbacks.ModelCheckpoint(filepath='models/best_model.keras',  # Save the model whenever the validation loss get better
                                                      monitor='val_loss',
                                                      save_best_only=True)]  # Ensure that only the best model is saved

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

results = model.fit(training_images, training_labels, batch_size=480, epochs=5, validation_split=0.2, callbacks=stopping_criterion)  # Training the network

model_summary = model.summary()  # Model Summary
model_config = model.get_config()  # Get a dictionary with the configuration of the model

dense_layer_weights = model.layers[1].get_weights()[0]  # Dense layer weights matrix
print(dense_layer_weights.shape)

bias = model.layers[1].get_weights()[1]  # Bias Vector
print(bias.shape)

test = model.predict(test_images)  # Query the model prediction
print(np.argmax(test[5]))
print(test_labels[5])

loss, acc = model.evaluate(test_images, test_labels)  # Performs model evaluation: Accuracy and Loss
print('Model loss', loss)
print('Model accuracy', acc)

# Plotting Loss x Epochs for training dataset
plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title('Accuracy/Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training', 'Validation'])
plt.show()

# Plotting Loss x Epochs for test dataset
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('Loss/Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Test', 'Validation'])
plt.show()

# model.save('model.keras')  # Save the model
# saved_model = load_model('model.keras')  # Load a model