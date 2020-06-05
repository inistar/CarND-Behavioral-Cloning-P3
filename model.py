import csv
import cv2
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from math import ceil
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D, Dropout

from tools import read_image_rgb, remove_noise, resize_image

samples = []
def read_csv_file(file, path):
    '''
    Read the CSV file and correct the image path. Add the center, left, and
    right images as image path and steering angle.
    '''
    count = 0
    with open(file) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            if count != 0:
                angle = float(line[3])
                
                # center image path correction
                center_image_path = line[0].split('/')[-1]
                center_image_path = path + center_image_path
                samples.append([center_image_path, angle])
                
                # left image path correction
                left_image_path = line[1].split('/')[-1]
                left_image_path = path + left_image_path
                samples.append([left_image_path, adjust_left_angle(angle)])
                
                # right image path correction
                right_image_path = line[2].split('/')[-1]
                right_image_path = path + right_image_path
                samples.append([right_image_path, adjust_right_angle(angle)])
                
            count += 1
            
    print("Total Images Added: {} x 3".format(count))
    return count

def flip_image(image, measurement):
    '''
    Flip the image to add more data.
    '''
    
    # steering angle greater than 0.15 and less than -0.15 are considered a turn
    if(abs(measurement) > 0.15):
        image       = cv2.flip(image, 1)
        measurement = measurement * -1.0
    return (image, measurement)

def adjust_left_angle(measurement):
    '''
    Add car recovery that is moving to far to the left. 
    '''
    # random number is generated between 0-0.5 and added to the constant and actual
    measurement = measurement + 0.2 + 0.5 * random.random()
    measurement = min(measurement, 1)
    return measurement
    
def adjust_right_angle(measurement):
    '''
    Add car recovery that is moving to far to the right. 
    '''
    # random number is generated between 0-0.5 and added to the constant and actual
    measurement = measurement - 0.2 - 0.5 * random.random()
    measurement = max(measurement, -1)
    return measurement

def generator(samples, batch_size=32):
    num_samples = len(samples)

    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            # for each batch, open the image file to RGB, remove any noise
            # with gaussian blur and resize the image to fit into neural network
            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = read_image_rgb(batch_sample[0])
                center_image = remove_noise(center_image)
                center_image = resize_image(center_image)


                measurement = float(batch_sample[1])

                images.append(center_image)
                angles.append(measurement)
                
                flipped_image, flipped_measurement = flip_image(center_image, measurement)
                images.append(flipped_image)
                angles.append(flipped_measurement)
                
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)



# Read the data of car going forward and backwards on the track
read_csv_file('./clean_data/driving_log.csv', './clean_data/IMG/')
read_csv_file('./clean_data_backwards/driving_log.csv', './clean_data_backwards/IMG/')

# Shuffle and split the sample into train and test data
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Use the train and validation to get a batch
batch_size=32
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Run NVIDIA Model
model = Sequential()

# Normalize the data
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(66,200,3)))

model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.25))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, \
            steps_per_epoch=ceil(len(train_samples)/batch_size), \
            validation_data=validation_generator, \
            validation_steps=ceil(len(validation_samples)/batch_size), \
            epochs=7, verbose=1)

model.save('model.h5')
