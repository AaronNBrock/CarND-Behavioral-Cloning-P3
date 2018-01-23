import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import random
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten, Cropping2D

INPUT_SHAPE = (160, 320, 3)

BATCH_SIZE = 16
VALID_SPLIT = 0.2
KEEP_PROB = 0.25
LEARNING_RATE = 0.0001
EPOCHS = 15

df = pd.read_csv('./data/driving_log.csv', names=['center', 'left', 'right', 'steer_angle', 'four', 'five', 'acceleration'])

x = df[['center', 'left', 'right']].values
y = df['steer_angle'].values

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=VALID_SPLIT)


def image_loader(data_images, data_steer_angles, batch_size, augment):

    i = 0
    while True:
        images = []
        steer_angles = []

        while len(images) < batch_size:
            steer_angle = data_steer_angles[i]

            if augment:
                # Randomly pick one of the three images
                rand = random.randint(0, 2)
                image_path = data_images[i][rand]
                if rand == 1:
                    steer_angle = steer_angle + 0.2
                elif rand == 2:
                    steer_angle = steer_angle - 0.2
            else:
                image_path = data_images[i][0]

            image = cv2.imread(image_path)

            if augment:
                # Randomly flip image
                if bool(random.getrandbits(1)):
                    image = cv2.flip(image, 1)
                    steer_angle = steer_angle*-1

            images.append(image)
            steer_angles.append(steer_angle)

            i += 1
            if i >= len(data_images):
                i = 0

        yield np.array(images), np.array(steer_angles)


model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=INPUT_SHAPE))
model.add(Lambda(lambda var: var/127.5-1.0))
model.add(Conv2D(24, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Dropout(KEEP_PROB))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
#model.summary()

model.compile(loss='mean_squared_error', optimizer=Adam(lr=LEARNING_RATE))

model.fit_generator(image_loader(x_train, y_train, BATCH_SIZE, True),
                    len(x_train),
                    EPOCHS,
                    max_q_size=1,
                    validation_data=image_loader(x_valid, y_valid, BATCH_SIZE, False),
                    nb_val_samples=len(x_valid))

model.save('model.h5')
