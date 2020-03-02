# from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
from sklearn.metrics import precision_recall_curve
import time

# ---------------------------------------------------------------- #
# INPUT
# ---------------------------------------------------------------- #

data_path = 'FlowerData'
test_images_indicies = list(range(300, 472))

# ---------------------------------------------------------------- #
# PARAMS
# ---------------------------------------------------------------- #

IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
TRAIN_BATCH_SIZE = 10
initial_epochs = 5
fine_tune_epochs = 10
VALIDATION_BATCH_SIZE = 20
TEST_BATCH_SIZE = len(test_images_indicies)
SHUFFLE_BUFFER_SIZE = 500
NEED_PRINT_PROCESS = True
base_learning_rate = 0.001
train_size = 240
validation_size = 60
fine_tune_at = 100
validation_steps = 2
test_steps = 5
start = time.time()
preferred_option = 2

# ---------------------------------------------------------------- #
# FUNCTIONS
# ---------------------------------------------------------------- #


def prepare_image(img):
    """
    takes an image and prepare it for nn
    :param img: image as nparray
    :return: resized image and its nn's representation
    """
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_for_nn = tf.cast(img, tf.float32)
    img_for_nn = (img_for_nn / 127.5) - 1
    img_for_nn = np.expand_dims(img_for_nn, axis=0)
    return img, img_for_nn


def format_images(images):
    """
    Does similar thing like the former function, but to the hole set of images
    """
    curr_images = []
    count = 0
    for image in images:
        count += 1
        img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        # cv2.imshow('name', img)
        # cv2.waitKey(1000)
        img = tf.cast(img, tf.float32)
        img = (img / 127.5) - 1
        curr_images.append(img)
        # print(count)
    return curr_images


def print_statement(statement):
    """
    Beautiful printing along the process.
    :param statement: string to print
    """
    if NEED_PRINT_PROCESS:
        print('[ARSENI & BAR]: %s' % statement)


def split_data(images, labels):
    """
    Splits the data and prepares it for nn
    :param images: preprocessed images
    :param labels: their labels
    :return: train_batches, validation_batches, test_dataset, test_labels
    """
    N = len(images)
    # print(N)
    train_images = []
    train_labels = []
    validation_images = []
    validation_labels = []
    test_images = []
    test_labels = []

    counter = 0
    for i in range(N):
        if i not in test_images_indicies:
            if counter < train_size:
                train_images.append(images[i])
                train_labels.append(labels[i])
            else:
                validation_images.append(images[i])
                validation_labels.append(labels[i])
        else:
            test_images.append(images[i])
            test_labels.append(labels[i])
        counter += 1

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_batches = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(TRAIN_BATCH_SIZE)

    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))
    validation_batches = validation_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(VALIDATION_BATCH_SIZE)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_dataset = test_dataset.batch(TEST_BATCH_SIZE)

    return train_batches, validation_batches, test_dataset, test_labels


def plot_worst_pics(curr_model):
    """
    Plots the worst pictures one by one.
    :param curr_model: the model
    """
    curr_images = obj['Data'][0]
    curr_labels = obj['Labels'][0]
    type1_errors = []
    type2_errors = []

    print()

    for i in test_images_indicies:
        _, img_for_nn = prepare_image(curr_images[i])
        predicted = curr_model.predict(img_for_nn)
        predicted = 1 / (1 + np.exp(predicted))
        # curr_label = 1 if curr_labels[i] == 0 else 0
        curr_label = curr_labels[i]
        predicted = 1 - predicted[0]
        if predicted < 0.5 and curr_label == 1:
            type1_errors.append((i, predicted))
        if predicted > 0.5 and curr_label == 0:
            type2_errors.append((i, predicted))

    type1_errors = sorted(type1_errors, key=lambda item: item[1], reverse=True)
    type2_errors = sorted(type2_errors, key=lambda item: item[1], reverse=False)

    for i in range(5):
        if i < len(type1_errors):
            index, curr_predicted = type1_errors[i]
            name = 'type: %s, error-index: %s, pic-index: %s, predicted score: %s' % ('I', i+1, index, curr_predicted)
            img, _ = prepare_image(curr_images[index])
            plt.title(name)
            plt.imshow(img)
            plt.show()

    for i in range(5):
        if i < len(type2_errors):
            index, curr_predicted = type2_errors[i]
            name = 'type: %s, error-index: %s, pic-index: %s, predicted score: %s' % ('II', i+1, index, curr_predicted)
            img, _ = prepare_image(curr_images[index])
            plt.title(name)
            plt.imshow(img)
            plt.show()


def plot_get_recalls_and_precisions(curr_model):
    """
    plots the recall-precision graph
    :param curr_model: the model
    :return:
    """

    curr_images = obj['Data'][0]
    curr_labels = obj['Labels'][0]
    predicted_array = []
    actual = []
    for i in test_images_indicies:
        _, img_for_nn = prepare_image(curr_images[i])
        actual.append(curr_labels[i])
        predicted = curr_model.predict(img_for_nn)
        predicted = 1 / (1 + np.exp(predicted))
        pred = 1 - predicted[0]
        predicted_array.append(pred)

    precision, recall, _ = precision_recall_curve(actual, predicted_array)
    plt.plot(recall, precision, 'c')
    plt.xlabel('recalls')
    plt.ylabel('precisions')
    plt.title('recall-precision graph')
    plt.show()


def plot_validation_vs_training(curr_history):
    """
    plots two graphs that showing the change of accuracy and loss in train and validation set
    :param curr_history: the history produced by model
    """
    acc = curr_history.history['accuracy']
    val_acc = curr_history.history['val_accuracy']

    loss = curr_history.history['loss']
    val_loss = curr_history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


print_statement('finished loading packages and functions')

# ---------------------------------------------------------------- #
# GET DATA
# ---------------------------------------------------------------- #

obj = sio.loadmat('%s/FlowerDataLabels.mat' % data_path)
images = obj['Data'][0]
labels = obj['Labels'][0]
print_statement('finished loading data')

# ---------------------------------------------------------------- #
# DATA REPRESENTATION
# ---------------------------------------------------------------- #

images = format_images(images)
labels = labels.tolist()
train_batches, validation_batches, test_dataset, test_labels = split_data(images, labels)

print_statement('finished data preparation')

# ---------------------------------------------------------------- #
# PREPARE MODEL
# ---------------------------------------------------------------- #

# Create the base model from the pre-trained model ResNet50V2
base_model = keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet', input_shape=IMG_SHAPE)
base_model.trainable = False  # without updeating the weights

# Let's take a look at the base model architecture - uncomment to see the summary
# base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = keras.layers.Dense(1)
model = keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate / 10),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

print_statement('the basic model:')
model.summary()

print_statement('finished primary model preparation')

# ---------------------------------------------------------------- #
# TRAIN
# ---------------------------------------------------------------- #

# TRAIN WITHOUT IMPROVEMENTS:

# loss0, accuracy0 = model.evaluate(test_batches, steps=validation_steps)
# print("initial loss: {:.2f}".format(loss0))
# print("initial accuracy: {:.2f}".format(accuracy0))
#
# history = model.fit(train_batches,
#                     epochs=initial_epochs,
#                     validation_data=validation_batches)

# ------------------------------------------------- #
# FINE TUNING
# ------------------------------------------------- #

# -------------------------- #
# OPTION 1 - best: batch-10, epochs-10
# -------------------------- #

if preferred_option == 1:
    print_statement('OPTION %s:' % preferred_option)

    # parameters check:
    if TRAIN_BATCH_SIZE != 10 or initial_epochs != 10:
        raise ValueError('The initial parameters are wrong.')

    base_model.trainable = True
    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate/10),
                  metrics=['accuracy'])

    print_statement('the chosen model:')
    model.summary()

    history = model.fit(train_batches,
                             epochs=initial_epochs,
                             validation_data=validation_batches)

# -------------------------- #
# OPTION 2 batches - 10, epochs - 5 or 7 or 10
# -------------------------- #

# ResNet101V2
if preferred_option == 2:
    print_statement('OPTION %s:' % preferred_option)

    # parameters check:
    if TRAIN_BATCH_SIZE != 10 or initial_epochs != 5:
        raise ValueError('The initial parameters are wrong.')

    base_model = keras.applications.resnet_v2.ResNet101V2(include_top=False, weights='imagenet', input_shape=IMG_SHAPE)
    base_model.trainable = False  # without updeating the weights

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = keras.layers.Dense(1)
    model = keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer
    ])

    base_model.trainable = True
    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])

    print_statement('the chosen model:')
    model.summary()
    history = model.fit(train_batches,
                        epochs=initial_epochs,
                        validation_data=validation_batches)

# -------------------------- #
# OPTION 3
# -------------------------- #

if preferred_option == 3:
    print_statement('OPTION %s:' % preferred_option)

    # parameters check:
    if TRAIN_BATCH_SIZE != 10 or initial_epochs != 10:
        raise ValueError('The initial parameters are wrong.')

    model = keras.Sequential()

    model.add(keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu',
                     input_shape=IMG_SHAPE))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(1))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate/10),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print_statement('the chosen model:')
    model.summary()
    history = model.fit(train_batches,
                        epochs=initial_epochs,
                        validation_data=validation_batches)

print_statement('finished training & tuning')

# ---------------------------------------------------------------- #
# TEST & EVALUATE
# ---------------------------------------------------------------- #

loss0, accuracy0 = model.evaluate(test_dataset, steps=1)
print("final loss: {:.2f}".format(loss0))
print("final accuracy: {:.2f}".format(accuracy0))
print("final test error: {:.2f}".format((1 - accuracy0)))
print_statement('finished evaluation & test')

# ---------------------------------------------------------------- #
# REPORT
# ---------------------------------------------------------------- #

print_statement('Reporting...')
# plot_worst_pics(model)
plot_get_recalls_and_precisions(model)
plot_validation_vs_training(history)

# ---------------------------------------------------------------- #
# MEASURING TIME
# ---------------------------------------------------------------- #

end = time.time()
print('minutes:', (end - start)/60)
