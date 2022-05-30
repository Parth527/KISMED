import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Activation
from tensorflow.keras.layers import Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard
from tensorflow.keras.optimizers import Adam
import pickle


def get_1DCNN(x_train, y_train, x_test, y_test, epochs=50):
    model = Sequential()

    model.add(Conv1D(16, 3, activation='relu', input_shape=(x_train.shape[1], 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Conv1D(32, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Conv1D(64, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))

    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])

    callback = [EarlyStopping(monitor='val_loss', patience=8),
                ModelCheckpoint(filepath='1DCNN_best_model.h5', monitor='val_loss', save_best_only=True)]

    # callback = [TensorBoard]

    history = model.fit(x_train, y_train,
                        batch_size=32,
                        epochs=epochs,
                        callbacks=callback,
                        validation_data=(x_test, y_test))

    return history, model
