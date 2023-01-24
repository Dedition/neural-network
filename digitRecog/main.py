import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# mnist = tf.keras.datasets.mnist  # 28x28 images of hand-written digits 0-9
# unpacks images to x_train/x_test and labels to y_train/y_test
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = tf.keras.utils.normalize(
#     x_train, axis=1)  # scales data between 0 and 1
# x_test = tf.keras.utils.normalize(
#     x_test, axis=1)  # scales data between 0 and 1

# model = tf.keras.models.Sequential()  # a basic feed-forward model
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # input layer
# model.add(tf.keras.layers.Dense(128, activation='relu'))  # hidden layer
# model.add(tf.keras.layers.Dense(128, activation='relu'))  # hidden layer
# model.add(tf.keras.layers.Dense(10, activation='softmax'))  # output layer

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[
#               'accuracy'])  # adam optimizer that minimizes the loss function

# model.fit(x_train, y_train, epochs=3)  # train the model

# model.save('handwritten.model')  # save the model

model = tf.keras.models.load_model('handwritten.model')

# loss, accuracy = model.evaluate(x_test, y_test)
# print(loss)
# print(accuracy)

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(
            f"digits/digit{image_number}.png")[:, :, 0]  # read the image
        img = np.invert(np.array([img]))  # invert the image
        prediction = model.predict(img)  # predict the image
        print("Prediction: ", np.argmax(prediction))  # print the prediction
        plt.imshow(img[0], cmap=plt.cm.binary)  # show the image
        plt.show()
    except:
        print("Error")
    finally:
        image_number += 1
