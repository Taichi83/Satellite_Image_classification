import time
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from subprocess import check_output
from datetime import timedelta
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils.datasets import load_EUROsatdata, onehot_encd, image_normalize_std
from utils.visualize import barplot_classes, sample_images
from utils.eval import eval_labels

from nets import keras_CNN

from keras.callbacks import TensorBoard
import keras.callbacks

# Parameters
SPLIT_RATE_TEST = 0.05
SPLIT_RATE_VALID = 0.01
RANDOM_STATE = 1
IMAGE_NORMALIZATION = True
EPOCHS = 10
DROPOUT = 0.5
BATCH_SIZE_TR = 256
 
def Keras_train_val(x_train, y_train, x_val, y_val, DROPOUT, BATCH_SIZE_TR, EPOCHS, 
                    SAVE_DIR = '../tmp/models/keras_cnn', LOG_DIR = '../tensorboard/keras_cnn', 
                    PLOT_DIR = '../tmp/figures/keras_cnn'):
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.isdir(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    model = keras_CNN.Keras_CNN(x_train.shape[1:], y_train.shape[1], DROPOUT)
    tbcallback = TensorBoard(log_dir=LOG_DIR, histogram_freq=1, write_graph=True, write_grads=True)
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE_TR, epochs=EPOCHS, verbose=1, 
          validation_data=(x_val, y_val), callbacks=[tbcallback])  
    
    # Plot training and validation loss
    acc = history.history['acc']
    acc_val = history.history['val_acc']
    loss = history.history['loss']
    loss_val = history.history['val_loss']
    epochs = range(len(acc))

    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(12,4))

    axL.plot(epochs, loss, linestyle='dashed', linewidth = 1.0, marker='o', label='Training loss')
    axL.plot(epochs, loss_val, color='orange', linestyle='solid', linewidth = 1.0, marker='o', label='Validation loss')
    axL.set_title('Training and validation loss')
    axL.legend()

    axR.plot(epochs, acc, linestyle='dashed', linewidth = 1.0, marker='o', label='Training accuracy')
    axR.plot(epochs, acc_val, color='orange', linestyle='solid', linewidth = 1.0, marker='o', label='Validation accuracy')
    axR.set_title('Training and validation accuracy')
    axR.legend()
    fig.savefig(os.path.join(PLOT_DIR, 'plot_loss_acc.png'), dpi=300)
    return model
    
# read data
data_path="../datasets/EuroSAT_RGB/"
images, labels = load_EUROsatdata(data_path)
print(images.shape, labels.shape)

# one-hot encoding
labels_oh, cl_nms = onehot_encd(labels)
print('labels:', cl_nms)
print('Dimensionality of one-hot encoded label file:', labels_oh.shape)

# save bar-plot
barplot_classes(labels_oh, cl_nms, save_path="../tmp/figures/keras_cnn", filename='class_dist.png')

# visualize sample images
sample_images(images, labels_oh, cl_nms, save_path="../tmp/figures/keras_cnn", filename='sample_images.png')

# Split training and test data
x_train, x_test, y_train, y_test = train_test_split(
        images, labels_oh, test_size=SPLIT_RATE_TEST, stratify=labels_oh, random_state=RANDOM_STATE)

# Normalization
if IMAGE_NORMALIZATION:
    x_train, x_test = image_normalize_std(x_train, x_test)

# Split training and validation data
x_train_tr, x_train_val, y_train_tr, y_train_val = train_test_split(
        x_train, y_train, test_size=SPLIT_RATE_VALID, stratify=y_train, random_state=RANDOM_STATE)

print("Size of:")
print("- Training-set:\t\t{}".format(x_train_tr.shape[0]))
print("- Validation-set:\t\t{}".format(x_train_val.shape[0]))
print("- Test-set:\t\t{}".format(x_test.shape[0]))
print("Dim of:")
print("- Training-set:\t\t{}".format(x_train_tr.shape[1:4]))
print("- Validation-set:\t\t{}".format(x_train_val.shape[1:]))
print("- Test-set:\t\t{}".format(x_test.shape[1:]))

model = Keras_train_val(x_train_tr, y_train_tr, x_train_val, y_train_val, DROPOUT, BATCH_SIZE_TR, EPOCHS,
                        SAVE_DIR = '../tmp/models/keras_cnn', LOG_DIR = '../tensorboard/keras_cnn', 
                        PLOT_DIR = '../tmp/figures/keras_cnn')

# Predict labels of test sets with trained model
pred_test = model.predict(x_test, verbose=1)
pred_test_le = np.argmax(pred_test, axis=1)
y_test_le = np.argmax(y_test, axis=1)

eval_labels(y_test_le, pred_test_le, save_path="../tmp/calc_log/keras_cnn", filename='confusion_matrix.csv')