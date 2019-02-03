# -*- coding: utf-8 -*-
"""
Last modified on Tuesday Feb 2 19:09 2019

@author: zhao-zh10
"""

import os
import csv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Cropping2D, Lambda, Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.callbacks import ModelCheckpoint, TensorBoard


# load data sample function
# load human drivers' behavior data, return the dataset sample list
# parameter description:
# data_file_list: input human drivers' behavior data file list
def load_data(data_file_list):
    samples = []
    angles_list_total = []
    angles_list_filtered = []
    for filename in data_file_list:
        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                # Filter out 90% of 0 angles to get more balanced dataset
                angle = float(line[3])
                angles_list_total.append(angle)
                if angle == 0 and np.random.uniform() <= 0.9:
                    continue
                samples.append(line)
                angles_list_filtered.append(angle)

    angle_data_total = np.array(angles_list_total)
    plt.figure()
    plt.hist(angle_data_total, bins=40, normed=1)
    plt.xlabel("Steering Angle")
    plt.ylabel("Frequency")
    plt.savefig("Angle_Data_Distribution_Total.png")
    angle_data_filtered = np.array(angles_list_filtered)
    plt.figure()
    plt.hist(angle_data_filtered, bins=40, normed=1)
    plt.xlabel("Steering Angle")
    plt.ylabel("Frequency")
    plt.savefig("Angle_Data_Distribution_Filtered.png")

    return samples


# sample generator function
# parameter description:
# whole_samples: the training or validation set sample list
# batch_size: model training hyperparameter batch_size
# steering_correction:  steering correction coefficient when using multiple cameras' images
def sample_generator(whole_samples, batch_size=32, steering_correction=0.2):
    num_samples = len(whole_samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(whole_samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = whole_samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                # Using images from the central camera
                center_image_name = os.path.join("./IMG/", os.path.split(batch_sample[0])[-1])
                center_image = mpimg.imread(center_image_name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                # Using Multiple Cameras
                # Left Camera
                left_image_name = os.path.join("./IMG/", os.path.split(batch_sample[1])[-1])
                left_image = mpimg.imread(left_image_name)
                left_angle = center_angle + steering_correction
                images.append(left_image)
                angles.append(left_angle)
                # Right Camera
                right_image_name = os.path.join("./IMG/", os.path.split(batch_sample[2])[-1])
                right_image = mpimg.imread(right_image_name)
                right_angle = center_angle - steering_correction
                images.append(right_image)
                angles.append(right_angle)

                # Data Augmentation: Flip the image left-right
                # Flipped Center Image
                center_image_flipped = np.fliplr(center_image)
                center_angle_flipped = -center_angle
                images.append(center_image_flipped)
                angles.append(center_angle_flipped)
                # Flipped Left Image
                left_image_flipped = np.fliplr(left_image)
                left_angle_flipped = -left_angle
                images.append(left_image_flipped)
                angles.append(left_angle_flipped)
                # Flipped Right Image
                right_image_flipped = np.fliplr(right_image)
                right_angle_flipped = -right_angle
                images.append(right_image_flipped)
                angles.append(right_angle_flipped)

            x_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(x_train, y_train)


# Get the behavior cloning convolutional neural network
# Reference Nvidia's CNN model for End-to-End self driving
# Add some dropout layers to avoid overfitting.
# Parameteer description:
# image_row: the height of camera's images; image_col: the width of camera's images
# image_channel: the color channel of camera's images;
# crop_top: the image cropping parameter, crop xx pixels from the top of the image
# crop_bottom: the image cropping parameter, crop xx pixels from the bottom of the image
# crop_left: the image cropping parameter, crop xx pixels from the left of the image
# crop_right: the image cropping parameter, crop xx pixels from the right of the image
def get_model(image_row, image_col, image_channel, crop_top, crop_bottom, crop_left, crop_right):
    # Referrence Nvidia Model
    model = Sequential()
    model.add(Cropping2D(cropping=((crop_top, crop_bottom), (crop_left, crop_right)),
                         input_shape=(image_row, image_col, image_channel)))
    model.add(Lambda(lambda x: x / 127.5 - 1.0))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    return model


# main function
def main():
    # image format(height:160,width:320,color channel:3)
    row, col, ch = 160, 320, 3
    # image cropping parameter
    crop_top, crop_bottom, crop_left, crop_right = 50, 20, 0, 0
    # model training hyperparameter batch_size
    batch_size = 256
    # model training hyperparameter epochs
    epochs = 30
    # steering correction coefficient when using multiple cameras' images
    steering_correction = 0.2
    # human drivers' behavior data file list (track1 and track2 are seperated)
    behavior_data_file_list = ["./driving_log_track1.csv", "./driving_log_track2.csv"]

    # load drivers' behavior data
    data_samples = load_data(behavior_data_file_list)
    # split training and validation sets
    train_samples, validation_samples = train_test_split(data_samples, test_size=0.2)
    # use the generator function to dynamically load training and validation sets
    train_generator = sample_generator(train_samples, batch_size=batch_size, steering_correction=steering_correction)
    validation_generator = sample_generator(validation_samples, batch_size=batch_size)
    # structure and compile the model
    model = get_model(image_row=row, image_col=col, image_channel=ch, crop_top=crop_top, crop_bottom=crop_bottom,
                      crop_left=crop_left, crop_right=crop_right)
    model.compile(loss="mse", optimizer="adam")
    # start training the model and using checkpoint to save the model weights
    filepath="model-{epoch:02d}-{val_loss:.4f}.h5"
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=False)
    # use TensorBoard to visualize the loss and val_loss
    callback_list = [checkpoint, TensorBoard(log_dir="./logs")]
    history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size,
                                         validation_data=validation_generator,
                                         validation_steps=len(validation_samples)/batch_size, epochs=epochs, verbose=1,
                                         callbacks=callback_list)
    # plot the training and validation loss for each epoch
    plt.figure()
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(["training set", "validation_set"], loc="upper right")
    plt.savefig('history.png')


if __name__ == "__main__":
    main()
