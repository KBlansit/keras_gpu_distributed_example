# load libraries
import socket
import keras
import numpy as np
import tensorflow as tf

from keras.datasets import mnist
from keras.models import Sequential
from keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D,\
    Dropout, Flatten, Dense
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import SGD

# deep learning parameters
BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 12
KERNEL_SIZE = (3, 3)
LEARN_RATE = 10e-4
MOMENTUM = 0.90
DECAY = LEARN_RATE/(EPOCHS)

# get computer name
PC = socket.gethostname()

# other parameters
USE_MULTI_GPU = True
IMG_ROWS, IMG_COLS = 28, 28
IMG_SHAPE = (IMG_ROWS, IMG_COLS, 1)

PREV_MODEL_PATH = "trained_model.hd"

# functions
def load_minst_data():
    """ loads mnist data
    :returns:
        x_train, y_train, x_test, y_test numpy arrays
    """
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # make into data
    x_train = x_train.reshape(x_train.shape[0], IMG_ROWS, IMG_COLS, 1)
    x_test = x_test.reshape(x_test.shape[0], IMG_ROWS, IMG_COLS, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    # return data
    return x_train, y_train, x_test, y_test

def make_datasets(x_input, y_input, batch_size=BATCH_SIZE, buffer=None):
    """ returns tensroflow dataset ready for distributed training
    :params:
        x_input: training numpy array
        y_input: training numpy array
        batch_size: batch size
        buffer_size: buffer size
    :returns:
        tensorflow dataset
    """
    # if we don't have set buffer size, use all x_input size
    if not buffer: buffer = x_input.shape[0]

    # return dataset
    return tf.data.Dataset.\
        from_tensor_slices((x_input, y_input)).\
        shuffle(buffer).\
        repeat().\
        batch(batch_size, drop_remainder=True)

def calculate_train_and_valid_steps(buffer_size, batch_size):
    """ calculates number of steps needed per batch
    :params:
        batch_size: batch size
        buffer_size: buffer size
    :returns:
        number of steps
    """
    # train number of steps
    if buffer_size % batch_size != 0:
        num_of_steps = buffer_size // batch_size + 1
    else:
        num_of_steps = buffer_size // batch_size

    # find ceiling
    num_of_steps = np.ceil(num_of_steps).astype('int')

    return num_of_steps

def load_model_with_scope(model_func):
    def wrapper(*args, **kwargs):
        # determine if we use multiple GPUs
        if PC == "AiDA-1" and USE_MULTI_GPU:
            # print infromation
            print("\n\nUsing multi GPU settings on: {}.\n\n".format(PC))

            # create a MirroredStrategy and open scope
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                # Everything that creates variables should be under the strategy scope
                # In general this is only model construction & `compile()`
                model = model_func(*args, **kwargs)

                print("Number of devices: {}.".format(strategy.num_replicas_in_sync))
        # not using multiple GPUs
        else:
            print("\n\nUsing single GPU settings on: {}.\n\n".format(PC))

            model = model_func(*args, **kwargs)
        return model
    return wrapper

def load_model_with_weights(prev_model_path):
    def wrap(model_func):
        def wrapper(*args, **kwargs):
            with tf.device('/cpu:0'):
                # load model
                prev_model = load_model(prev_model_path)
                prev_weights = prev_model.get_weights()

                # clean up so we don't overallocate space
                del prev_model

            # load model and set weights
            model = model_func(*args, **kwargs)
            model.set_weights(prev_weights)

            # message
            print("Set model weights")

            # return
            return model
        return wrapper
    return wrap

def cnn_model():
    """ loads a simple cnn model
    :returns:
        Keras model
    """
    # define model
    # conv layers
    inputs = Input(IMG_SHAPE, name = "input")
    conv_1 = Convolution2D(32, KERNEL_SIZE, padding="same", activation="relu",\
        name="conv_1")(inputs)
    conv_2 = Convolution2D(64, KERNEL_SIZE, padding="same", activation="relu",\
        name="conv_2")(conv_1)
    max_pool = MaxPooling2D(pool_size=(2, 2), name="Maxpooling")(conv_2)

    # dense and dropout layers
    drop_1 = Dropout(0.25, name="drop_1")(max_pool)
    flat = Flatten(name="flat")(drop_1)
    dense_1 = Dense(128, name="dense_1")(flat)
    drop_2 = Dropout(0.5, name="drop_2")(dense_1)
    out = Dense(NUM_CLASSES, activation='softmax', name="out")(drop_2)

    # define model
    model = Model(inputs=inputs, outputs=out)

    # compile model
    model.compile(
        loss=categorical_crossentropy,
        optimizer=SGD(
            lr=float(LEARN_RATE),
            decay=float(DECAY),
            momentum=MOMENTUM,
        ),
        metrics=['accuracy'],
    )

    # return model
    return model


# load data
x_train, y_train, x_test, y_test = load_minst_data()


# make into tensorflow datasets
# we can use size of datasets since it's <2Gbs
train_buffer_size = x_train.shape[0]
test_buffer_size = x_test.shape[0]

# make datasets
train_dataset = make_datasets(x_train, y_train)
test_dataset = make_datasets(x_test, y_test)

# calculate number of steps
train_parallel_steps = calculate_train_and_valid_steps(
        train_buffer_size,
        BATCH_SIZE,
)
test_parallel_steps = calculate_train_and_valid_steps(
        test_buffer_size,
        BATCH_SIZE,
)


# WILL NOT WORK
# make a learning strategy and open scope for compilling model
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}.".format(strategy.num_replicas_in_sync))
with strategy.scope():
    model = load_model(PREV_MODEL_PATH)


# WILL WORK
# load model with cpu
with tf.device('/cpu:0'):
    # load model
    prev_model = load_model(PREV_MODEL_PATH)
    prev_weights = prev_model.get_weights()


# make a learning strategy and open scope for
# compiling model
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}.".format(strategy.num_replicas_in_sync))
with strategy.scope():
    model = cnn_model()
model.set_weights(prev_weights)

# get score and print
score = model.evaluate(test_dataset, steps=test_parallel_steps)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
