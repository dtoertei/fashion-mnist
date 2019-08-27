from keras import optimizers
from keras.datasets import fashion_mnist
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.constraints import max_norm
from sklearn.model_selection import StratifiedShuffleSplit
from numpy.random import seed
from tensorflow import set_random_seed

from CNNmodel import *
from train_callbacks import *
from utils import *

seed(1)
set_random_seed(2)


# Fashion MNIST related variables
img_height = 28
img_width = 28
num_channels = 1
num_classes = 10
labelNames = ["top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]

# CNN related variables and setting
K.set_image_dim_ordering('tf')
input_shape = (img_height, img_width, num_channels)


if __name__ == '__main__':

    ((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()

    # explore Fashion-MNIST
    #occurenceHistogram(trainY)
    #plt.imshow(trainX[100])

    datagen, trainX, testX = preprocessData(trainX, testX)
    trainX = trainX.reshape(trainX.shape[0], img_height, img_width, num_channels)
    testX = testX.reshape(testX.shape[0], img_height, img_width, num_channels)
    # if std_normalization is set
    datagen.fit(trainX)

    # stratified k-fold train/validate split generator
    k_fold = 3
    train_idx = []
    val_idx = []
    stratgen = StratifiedShuffleSplit(n_splits=k_fold, test_size=0.2, random_state=0)
    for idx1, idx2 in stratgen.split(np.zeros(len(trainY)), trainY):
        train_idx.append(idx1)
        val_idx.append(idx2)

    # train related variables: stratified k-fold
    fold_epochs = 30
    batch_size = 128
    LR = 0.001
    norm = max_norm(4.0)
    linearValLR = 0.5
    minValLR = 1e-8
    patienceLR = 2

    # apply weights for classes
    class_weight = {0: 1,
                    1: 1,
                    2: 1,
                    3: 1,
                    4: 1,
                    5: 1,
                    6: 1,
                    7: 1,
                    8: 1,
                    9: 1}

    max_val_acc = [0, 0, 0]

    for i in range(k_fold):

        # prepare fold data
        trainFold_X = trainX[train_idx[i]]
        valFold_X = trainX[val_idx[i]]
        trainFold_Y = to_categorical(trainY[train_idx[i]], num_classes)
        valFold_Y = to_categorical(trainY[val_idx[i]], num_classes)

        # checkpoint_path = './weightsInception{:d}'.format(i) + '-{epoch:02d}-{val_acc:.2f}.hdf5'
        checkpoint_path = './weights_CNN{:d}'.format(i) + '.hdf5'
        checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True,
                                              mode='max')

        # enlist callbacks
        printLRval_callback = LRval()
        reduceLR_callback = ReduceLROnPlateau(monitor='val_loss', factor=linearValLR, patience=patienceLR, min_lr=minValLR)
        callbacks_list = [printLRval_callback, reduceLR_callback, checkpoint_callback]
        histories_callback = [] # for plotting learning curves

        # here we got stratified train and validate data in correct format for keras's fit()
        print('Fold {:d} is being run'.format(i))
        model = conv2Dsimple2(input_shape, num_classes, norm)
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=LR), metrics=['accuracy'])
        # model.summary()
        histories_callback.append(model.fit_generator(datagen.flow(trainFold_X, trainFold_Y, batch_size=batch_size),
                                                      steps_per_epoch=len(trainFold_X) / batch_size,
                                                      epochs=fold_epochs,
                                                      validation_data=(valFold_X, valFold_Y),
                                                      callbacks=callbacks_list,
                                                      verbose=1,
                                                      class_weight=class_weight))

        # plots train/val loss and accuracy
        plotLearningCurves(i, histories_callback)

        # evaluate current fold model on its validation set
        eval = model.evaluate(trainX[val_idx[i]], valFold_Y, verbose=0)
        print('Validation loss:', eval[0])
        print('Validation accuracy:', eval[1])

        if (eval[1] > max_val_acc[i]):
            max_val_acc[i] = eval[1]

        # confusion matrix, precision, recall and f1 score
        testCNN(model, valFold_X, trainY[val_idx[i]], labelNames)


    #evaluate best fold on Fashion MNIST test set
    print('Best fold was fold no.: ', np.argmax(max_val_acc))
    model.load_weights('./weights_CNN{:d}.hdf5'.format(np.argmax(max_val_acc)))
    print('Best model metrics: \n')
    testCNN(model, testX, testY, labelNames)
    eval = model.evaluate(testX, to_categorical(testY), verbose=0)
    print('Test set loss:', eval[0])
    print('Test set accuracy:', eval[1])
    # plot model architecture
    if isinstance(model, Sequential):
        plot_model(seq2funcModel(model), to_file='CNN_arch.png', show_shapes=True, show_layer_names=False)
    else:
        plot_model(model, to_file='CNN_arch.png', show_shapes=True, show_layer_names=False)

