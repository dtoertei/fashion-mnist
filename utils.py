import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# verify number of samples with classes binning
def occurenceHistogram(trainY):
    #occurencies, classes = np.histogram(trainY,bins=np.arange(11))
    # or plot
    _ = plt.hist(trainY, bins='auto')
    plt.title('Fashion MNIST trainY histogram')
    plt.xlabel('classes num')
    plt.ylabel('num occurencies')
    plt.show()

def preprocessData(trainX, testX):
    # Data augmentation
    datagen = ImageDataGenerator(
    #    featurewise_center=True,
    #    featurewise_std_normalization=True
    #    rotation_range=10,
    #    width_shift_range=0.15,
    #    height_shift_range=0.15,
    #    horizontal_flip=True,
    #    fill_mode='nearest'
    )
    # scaling
    trainX = trainX/255
    testX = testX/255

    return datagen, trainX, testX


# plot the training loss and accuracy
def plotLearningCurves(kfold, histories_callback):

    fold_epochs = len(histories_callback[0].history["acc"])
    # plot losses
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(range(0, fold_epochs), histories_callback[0].history["loss"], label="train_loss")
    plt.plot(range(0, fold_epochs), histories_callback[0].history["val_loss"], label="val_loss")
    plt.title('Train/Val Losses of fold {:d}'.format(kfold))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('LossPlotFold_{:d}_InceptionCNN.png'.format(kfold))
    # plot accuracies
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(range(0, fold_epochs), histories_callback[0].history["acc"], label="train_acc")
    plt.plot(range(0, fold_epochs), histories_callback[0].history["val_acc"], label="val_acc")
    plt.title('Train/Val Accuracies of fold {:d}'.format(kfold))
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig('AccPlotFold_{:d}_InceptionCNN.png'.format(kfold))


# see model performance and actual repartition of misses
# Y contains [0..9]
def testCNN(model, X, Y, labelNames):

    # get the predictions for the test data
    if isinstance(model, Sequential):
        predicted_classes = model.predict_classes(X)
    else:
        predicted_classes = model.predict(X)
        predicted_classes = np.argmax(predicted_classes, axis=1)

    print(classification_report(Y, predicted_classes, target_names=labelNames))
    print(confusion_matrix(Y, predicted_classes))

# convert model from sequential to functional
def seq2funcModel(seqModel):

    input_layer = layers.Input(batch_shape=seqModel.layers[0].input_shape)
    prev_layer = input_layer
    for layer in seqModel.layers:
        prev_layer = layer(prev_layer)

    funcmodel = models.Model([input_layer], [prev_layer])

    return funcmodel
