from keras.models import Sequential, Model
from keras.layers import MaxPooling2D, Conv2D, Dropout, Flatten, Dense, Input, concatenate
from keras.layers.normalization import BatchNormalization
from keras.initializers import glorot_uniform


def inceptionCNN(input_shape, num_classes, norm=None):
    X = Input(input_shape)

    X1 = Conv2D(16, (1, 1), padding = 'same', activation='relu', kernel_initializer = glorot_uniform(seed=0),
                kernel_constraint=norm, bias_constraint=norm)(X)
    X1 = Conv2D(32, (3, 3), padding = 'same', activation='relu', kernel_initializer = glorot_uniform(seed=0),
                kernel_constraint=norm, bias_constraint=norm)(X1)

    X2 = Conv2D(16, (1, 1), padding = 'same', activation='relu', kernel_initializer = glorot_uniform(seed=0),
                kernel_constraint=norm, bias_constraint=norm)(X)
    X2 = Conv2D(32, (5, 5), padding = 'same', activation='relu', kernel_initializer = glorot_uniform(seed=0),
                kernel_constraint=norm, bias_constraint=norm)(X2)

    X3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(X)
    X3 = Conv2D(32, (1,1), padding='same', activation='relu', kernel_initializer = glorot_uniform(seed=0),
                kernel_constraint=norm, bias_constraint=norm)(X3)

    X4 = Conv2D(32, (1, 1), padding = 'same', activation='relu', kernel_initializer = glorot_uniform(seed=0),
                kernel_constraint=norm, bias_constraint=norm)(X)

    concat_1 = concatenate([X1, X2, X3, X4], axis = -1)
    flatten_1 = Flatten()(concat_1)

    dense_1 = Dense(200, activation='relu', kernel_constraint=norm, bias_constraint=norm)(flatten_1)
    dropout = Dropout(0.25)(dense_1)
    dense_2 = Dense(100, activation='relu', kernel_constraint=norm, bias_constraint=norm)(dropout)
    dropout2 = Dropout(0.25)(dense_2)
    output = Dense(num_classes, activation='softmax', kernel_initializer = glorot_uniform(seed=0),
                   kernel_constraint=norm, bias_constraint=norm)(dropout2)

    return Model(X, output)


def conv2Dnet(input_shape, num_classes, norm=None):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding = 'same', activation ='relu', kernel_initializer = glorot_uniform(seed=0),
                     input_shape=input_shape, kernel_constraint=norm, bias_constraint=norm))
    model.add(MaxPooling2D((2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu', kernel_initializer = glorot_uniform(seed=0), kernel_constraint=norm,
                    bias_constraint=norm))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation = 'softmax', kernel_initializer = glorot_uniform(seed=0),
                    kernel_constraint=norm, bias_constraint=norm))

    return model


def conv2Dsimple(input_shape, num_classes, norm=None):
    model = Sequential()
    model.add(Conv2D(16, (1, 1), padding = 'same', activation ='relu', kernel_initializer = glorot_uniform(seed=0),
                     input_shape=input_shape, kernel_constraint=norm, bias_constraint=norm))
    model.add(Conv2D(32, (3, 3), padding = 'same', activation ='relu', kernel_initializer = glorot_uniform(seed=0),
                     kernel_constraint=norm, bias_constraint=norm))
    model.add(MaxPooling2D((4,4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu', kernel_initializer = glorot_uniform(seed=0), kernel_constraint=norm,
                    bias_constraint=norm))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation = 'softmax', kernel_initializer = glorot_uniform(seed=0),
                    kernel_constraint=norm, bias_constraint=norm))

    return model


def conv2Dsimple2(input_shape, num_classes, norm=None):
    model = Sequential()
    model.add(Conv2D(16, (1, 1), padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=0),
                     input_shape=input_shape, kernel_constraint=norm, bias_constraint=norm))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=0),
                     kernel_constraint=norm, bias_constraint=norm))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_initializer=glorot_uniform(seed=0),
                     kernel_constraint=norm, bias_constraint=norm))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_initializer=glorot_uniform(seed=0), kernel_constraint=norm,
                    bias_constraint=norm))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=glorot_uniform(seed=0),
                    kernel_constraint=norm, bias_constraint=norm))

    return model


def conv2Dsimple3(input_shape, num_classes, norm=None):
    model = Sequential()
    model.add(Conv2D(16, (1, 1), padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=0),
                     input_shape=input_shape, kernel_constraint=norm, bias_constraint=norm))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=0),
                     kernel_constraint=norm, bias_constraint=norm))
    model.add(MaxPooling2D((2, 2), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (1, 1), padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=0),
                     kernel_constraint=norm, bias_constraint=norm))
    model.add(Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_initializer=glorot_uniform(seed=0),
                     kernel_constraint=norm, bias_constraint=norm))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(300, activation='relu', kernel_initializer=glorot_uniform(seed=0), kernel_constraint=norm,
                    bias_constraint=norm))
    model.add(Dropout(0.25))
    model.add(Dense(150, activation='relu', kernel_initializer=glorot_uniform(seed=0), kernel_constraint=norm,
                    bias_constraint=norm))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=glorot_uniform(seed=0),
                    kernel_constraint=norm, bias_constraint=norm))

    return model


