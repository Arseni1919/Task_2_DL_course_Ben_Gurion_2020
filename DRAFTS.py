import numpy as np
import matplotlib.pyplot as plt
x1 = np.arange(3.0)
print(x1)
x2 = np.arange(3.0)
print(x2)
print(np.multiply(x1, x2))

plt.plot(x1, x2)
plt.plot([10,10,20], [20,30,40], 'c')
plt.show()


# def getKey(item):
#     return item[1]
plt.plot()


def plot_get_recalls_and_precisions(curr_model, curr_test_dataset, curr_test_labels):
    """
    plots the recall-precision graph.
    :param curr_model:
    :param curr_test_dataset:
    :param curr_test_labels:
    :return:
    """
    # actual = np.asarray(curr_test_labels)
    # predicted = curr_model.predict(curr_test_dataset)
    # predicted = 1 / (1 + np.exp(predicted))
    # predicted = predicted.reshape(actual.shape)

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
    # ----------------------------------------------------------- #
    precision, recall, _ = precision_recall_curve(actual, predicted_array)
    plt.plot(recall, precision, 'c')
    plt.xlabel('recalls')
    plt.ylabel('precisions')
    plt.title('recall-precision graph')
    plt.show()
    # ----------------------------------------------------------- #
    # x, y = [], []
    # not_actual = np.zeros(actual.shape)
    # not_actual[actual == 1] = 0
    # not_actual[actual == 0] = 1
    #
    #
    # for t in np.linspace(0, 1, 100):
    #     curr_predicted = np.zeros(predicted.shape)
    #     not_curr_predicted = np.zeros(predicted.shape)
    #     curr_predicted[predicted > t] = 1
    #     curr_predicted[predicted < t] = 0
    #     not_curr_predicted[curr_predicted == 1] = 0
    #     not_curr_predicted[curr_predicted == 0] = 1
    #
    #     multiplication = np.multiply(curr_predicted, actual)
    #     TP = np.count_nonzero(np.multiply(curr_predicted, actual))
    #     # TN = np.count_nonzero(np.multiply(not_curr_predicted, not_actual))
    #     FP = np.count_nonzero(np.multiply(curr_predicted, not_actual))
    #     FN = np.count_nonzero(np.multiply(not_curr_predicted, actual))
    #     if (TP + FP) == 0 or (TP + FN) == 0:
    #         continue
    #     recall = TP / (TP + FN)
    #     precision = TP / (TP + FP)
    #     x.append(recall)
    #     y.append(precision)
    #
    # plt.plot(x, y)
    # plt.xlabel('recalls')
    # plt.ylabel('precisions')
    # plt.title('recall-precision graph')
    # plt.show()


# show images in two separate windows
# for image, label in zip(images, labels):
#     name = 'flower' if label == 1 else 'not flower'
#     cv2.imshow(name, image)
#     print(type(image), image.dtype)
#     cv2.waitKey(1000)

# for image_batch, label_batch in train_dataset.take(1):
#     print('image_batch.shape: ', image_batch.shape)
#     feature_batch = base_model(image_batch)
#     print('feature_batch.shape:', feature_batch.shape)
#     global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
#     feature_batch_average = global_average_layer(feature_batch)
#     print('feature_batch_average.shape:', feature_batch_average.shape)
#     prediction_layer = keras.layers.Dense(1)
#     prediction_batch = prediction_layer(feature_batch_average)
#     print('prediction_batch.shape:', prediction_batch.shape)
