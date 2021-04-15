from cosc420_assig1_data import load_data
import show_methods
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os, pickle, gzip, json

# Define the task/type here
Task = 'simple'
DataType = 'clean'
Epochs = 1

# Saving the figures rather than showing them is used for remote work.
SaveFigures = True
LoadFromFile = False

# Loading data from supplied functions
# Training images are a 20000x28x28x3 matrix - 20k images, 28x28 pixels of RGB
# Training labels are a 20000 vector of indexes for class names
# Test images & labels are the same, but for only 4000 images
(train_images, train_labels), (test_images, test_labels), class_names = load_data(task=Task, dtype=DataType)
n_classes = len(class_names)

# Show a sample of collected training data
show_methods.show_data_images(images=train_images[:16],labels=train_labels[:16],class_names=class_names)
if SaveFigures:
   plt.savefig(Task + DataType + '_train.png')

# Perpare save filename/directory
save_name = os.path.join('saved', Task + DataType)
net_save_name = save_name + '_fc_net.h5'
history_save_name = save_name + '_fc_net.hist'
if not os.path.isdir("saved"):
   os.mkdir('saved')

# Load previously trained neural network model
if LoadFromFile:
   print("Loading neural network from %s..." % net_save_name)
   net = tf.keras.models.load_model(net_save_name)

   # Load the training history
   with gzip.open(history_save_name) as f:
     history = pickle.load(f)

# Create and train a neural network model
else:
   # Pre-Define layers in an array structure for easier recording later
   layers = [
      tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1), activation='relu', input_shape=(28, 28, 3)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(units=n_classes,activation='softmax')
   ]
   net = tf.keras.models.Sequential(layers)

   # Define training regime: type of optimiser, loss function to optimise and type of error measure to report
   net.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

   # Train the model for 20 epochs, using 33% of the train data as validation
   train_info = net.fit(train_images, train_labels, validation_split=0.33, epochs=Epochs)

   # Save model to file
   print("Saving neural network to %s..." % net_save_name)
   net.save(net_save_name)

   # Save training history to file
   history = train_info.history
   with gzip.open(history_save_name, 'w') as f:
      pickle.dump(history, f)

# Plot training and validation accuracy over the course of training
plt.figure()
plt.plot(history['acc'], label='accuracy')
plt.plot(history['val_acc'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

if SaveFigures:
   plt.savefig(Task + DataType + '_history.png')
plt.show()

# Evaluating the neural network model within tensorflow
loss_train, accuracy_train = net.evaluate(train_images,  train_labels, verbose=0)
loss_test, accuracy_test = net.evaluate(test_images, test_labels, verbose=0)
print("Train accuracy (tf): %.2f" % accuracy_train)
print("Test accuracy  (tf): %.2f" % accuracy_test)

# Compute output for 16 test images
y_hat_test = net.predict(test_images[:16])
y_hat_test = np.argmax(y_hat_test, axis=1)

# Show true labels and predictions for 16 test images
show_methods.show_data_images(images=test_images[:16],
                              labels=test_labels[:16],predictions=y_hat_test,
                              class_names=class_names)
if SaveFigures:
   plt.savefig(Task + DataType + '_results.png')
plt.show()

# Append layer structure to csv
# j = open(Task + DataType + '_structure_history.json', 'a')
# j.write(str(json.dumps([l.name for l in layers])))