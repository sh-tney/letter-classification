from cosc420_assig1_data import load_data
import show_methods
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os, pickle, gzip, csv, inspect

# Define the task/type here
Task = 'simple'
DataType = 'clean'
Epochs = 50

# Saving the figures rather than showing them is used for remote work.
SaveFigures = False
LoadFromFile = True
Iteration = "C0Dupe42FineGrainNoisyFinal" # Runs model based on the final FineGrain structure 
# Iteration = "C1FinalSimpleClean" # Runs a model based on the structure in this file

# Loading data from supplied functions
# Training images are a 20000x28x28x3 matrix - 20k images, 28x28 pixels of RGB
# Training labels are a 20000 vector of indexes for class names
# Test images & labels are the same, but for only 4000 images
(train_images, train_labels), (test_images, test_labels), class_names = load_data(task=Task, dtype=DataType)
n_classes = len(class_names)

# Separate out a training set
n = len(train_images)
n_valid = 5000 # 1/4 20,000
np.random.seed(0)
perm = np.random.permutation(n)

i_valid = perm[:n_valid]
valid_images = train_images[i_valid]
valid_labels = train_labels[i_valid]

i_train = perm[n_valid:]
train_images = train_images[i_train]
train_labels = train_labels[i_train]

# Perpare save filename/directory
save_name = os.path.join('saved' + Iteration, Task + DataType)
net_save_name = save_name + '_fc_net.h5'
history_save_name = save_name + '_fc_net.hist'
if not os.path.isdir("saved" + Iteration):
   os.mkdir('saved' + Iteration)

# Show a sample of collected training data
show_methods.show_data_images(images=train_images[:16],labels=train_labels[:16],class_names=class_names)

# Load previously trained neural network model
if LoadFromFile:
   net = tf.keras.models.load_model(net_save_name)

   # Load the training history
   with gzip.open(history_save_name) as f:
     history = pickle.load(f)

# Create and train a neural network model
else:
   # Save only the best weights for validation accuracy, run no more than half Epochs if no improvement
   save_best = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=Epochs/4, restore_best_weights=True)

   # Pre-Define layers in an array structure for easier recording later
   layers = [
      (lambda: tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', input_shape=(28, 28, 3), padding='same')),
      (lambda: tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))),
      (lambda: tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same')),
      (lambda: tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))),
      (lambda: tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same')),
      (lambda: tf.keras.layers.BatchNormalization()),
      (lambda: tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))),
      (lambda: tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', padding='same')),
      (lambda: tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same')),
      (lambda: tf.keras.layers.BatchNormalization()),
      (lambda: tf.keras.layers.Flatten()),
      (lambda: tf.keras.layers.Dropout(0.45)),
      (lambda: tf.keras.layers.Dense(units=n_classes,activation='softmax'))
   ]
   net = tf.keras.models.Sequential(list(map(lambda l: l(), layers)))

   # Define training regime: type of optimiser, loss function to optimise and type of error measure to report
   net.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

   # Augmentation
   imgGen = tf.keras.preprocessing.image.ImageDataGenerator(
       zca_epsilon=1e-06,
       width_shift_range = 0.1,
       height_shift_range = 0.1,
       fill_mode='nearest'
   )
   imgGen.fit(train_images)
   aug = imgGen.flow(train_images, train_labels)

   # Train the model for defined number of epochs, using 25% of the train data as validation
   # train_info = net.fit(train_images, train_labels, validation_split=0.25, epochs=Epochs, callbacks=[save_best]) # Non-augmentation code
   train_info = net.fit(aug, validation_data=(valid_images, valid_labels), shuffle=True, epochs=Epochs, callbacks=[save_best])

   # Save model to file
   net.save(net_save_name)

   # Save training history to file
   history = train_info.history
   with gzip.open(history_save_name, 'w') as f:
      pickle.dump(history, f)

# Establishing these variables to avoid conflicts between tf/keras versions on different environments
acc = None
val_acc = None
if('acc' in history):
   acc = history['acc']
   val_acc = history['val_acc']
else:
   acc = history['accuracy']
   val_acc = history['val_accuracy']

# Plot training and validation accuracy over the course of training
plt.figure()
plt.plot(acc, label='accuracy')
plt.plot(val_acc, label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

if SaveFigures:
   plt.savefig(save_name + '_history.png')
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
                              labels=test_labels[:16],
                              predictions=y_hat_test,
                              class_names=class_names)
if SaveFigures:
   plt.savefig(save_name + '_results.png')
plt.show()

# Append layer structure to csv
if not LoadFromFile:
   with open(Task + DataType + '_structure_history.csv', 'a') as c:
      w = csv.writer(c)
      w.writerow([Iteration, accuracy_test, accuracy_train, Epochs] + list(map(lambda l: inspect.getsource(l).strip(), layers)))