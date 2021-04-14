from cosc420_assig1_data import load_data

# Define the task/type here
Task = 'simple'
DataType = 'clean'

# Loading data from supplied functions
# Training images are a 20000x28x28x3 matrix - 20k images, 28x28 pixels of RGB
# Training labels are a 20000 vector of indexes for class names
(train_images, train_labels), (test_images, test_labels), class_names = load_data(task=Task, dtype=DataType)