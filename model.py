import numpy as np
import glob
import cv2
import re
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
import keras
from tensorflow.keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import classification_report


def define_model():
    model = Sequential()
    model.add(Conv2D(filters=4, kernel_size=(5, 5), strides=(3, 3),
                     activation='relu', input_shape=(128, 128, 1)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(kernel_size=(5, 5), strides=(2, 2),
                     filters=4, activation='relu', input_shape=(21, 21, 1)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Flatten())
    model.add(Dense(units=4, activation='tanh'))
    model.add(Dense(2, activation='softmax'))
    print('\nCNN model : ')
    model.summary()
    return model


def compile_fit_model(model):
    print('\nTraining the dataset : ')
    OPTIMIZER = SGD(learning_rate=0.1)
    x_train_gs = np.load(
        '../input/skincancerdetectiondcnn/train_dataset/x_train.npy')
    y_train_gs = np.load(
        '../input/skincancerdetectiondcnn/train_dataset/y_train.npy')
    x_train, x_val, y_train, y_val = train_test_split(x_train_gs, y_train_gs, test_size=0.2,
                                                      random_state=15)
    model.compile(optimizer=OPTIMIZER,
                  loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    model_fit = model.fit(x=x_train, y=y_train, epochs=50,
                          batch_size=32, validation_data=(x_val, y_val), verbose=1)

    print('Graph of epochs vs loss')
    %matplotlib inline
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(model_fit.history['loss'])
    plt.plot(model_fit.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.figure()
    plt.show()
    return model


def plot_normalized_confusion_matrix(cm, classes,
                                     normalize=False,
                                     title='Confusion matrix',
                                     cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, with normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=55)
    plt.yticks(tick_marks, classes)
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def main():
    print('Training dataset : ')
    print('     No. of images : 2000')
    print('     Input size to the CNN model : 128*128')
    print('Testing dataset : ')
    print('     No. of images : 120')
    print('     Input size to the CNN model : 128*128')
    model = define_model()
    model = compile_fit_model(model)
    model.save("./model.h5")
    x_test = np.load(
        '../input/skincancerdetectiondcnn/test_dataset/x_test.npy')
    y_test = np.load(
        '../input/skincancerdetectiondcnn/test_dataset/y_test.npy')
    print('\nEvalation with testing dataset: ')
    score = model.evaluate(x_test, y_test, verbose=2)
    print('Test accuracy : ', score[1]*100, '%')
    cm_plot_label = ['benign', 'malignant']  # 0 - benign, 1 - malignant
    y_predict = model.predict(x_test).argmax(axis=1)
    cm = confusion_matrix(y_test.argmax(axis=-1), y_predict, normalize='true')
    plot_normalized_confusion_matrix(
        cm, cm_plot_label, title='Normalized Confusion Matrix')
    print(classification_report(y_test.argmax(axis=-1), y_predict))


if __name__ == "__main__":
    main()
