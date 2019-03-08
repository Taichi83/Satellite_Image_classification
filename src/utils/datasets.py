import time
import os
import numpy as np
from tqdm import tqdm
import cv2

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

def load_EUROsatdata(data_path="../datasets/EuroSAT_RGB/"):
    """Loads EUROsat datasets from 'data_path'
    Converts JPEG format files to numpy array
    Describs class labels to numpy array
    Args:
        data_path: String path to a folder containing subfolders of images.
    Returns:
        images: Numpy array of images [-1, hight, width, dims]
        labels: Numpy array of labels [-1,]
    """
    # print(os.getcwd())
    cl_nms = os.listdir(data_path)
    # Eliminate hidden file's name
    cl_nms = [cl_nms[i] for i in range(len(cl_nms)) if (cl_nms[i][0])!='.']
    images = []
    labels = []
    for dn in range(len(cl_nms)):
        cl_nm = cl_nms[dn]
        file_nm_list = os.listdir(os.path.join(data_path, cl_nm))
        for fn in tqdm(range(len(file_nm_list))):
            im = cv2.imread(os.path.join(os.path.join(data_path, cl_nm), file_nm_list[fn]))
            images.append(im)
            labels.append(cl_nm)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

## one-hot encoding
def onehot_encd(labels):
    """Converts label encodrding to one-hot encoding
    Args:
        images: Numpy array of images [-1, hight, width, dims]
    Returns:
        labels_oh: Binary array in Numpy array format [-1, number of classes]
        cl_nms: Numpy array of strings; class names [number of classes]
    """
    cl_nms = np.unique(labels)
    le = preprocessing.LabelEncoder()
    le.fit(cl_nms)
    labels_enc = le.transform(labels).reshape(-1,1)
    labels_oh = OneHotEncoder(categories='auto').fit_transform(labels_enc).A
    return labels_oh, cl_nms

def image_normalize_std(x_train, x_test):
    """Normalization of each pixel value based on average and standard deviation of training dataset
    Args:
        x_train: Numpy array of training images [-1, hight, width, dims]
        x_test: Numpy array of test images [-1, hight, width, dims]
    Returns:
        np.divide((x_train - x_mean), x_std): Normalized training data
        np.divide((x_test - x_mean), x_std): Normalized test data
    """
    x_mean = np.mean(x_train, axis=0)
    x_std = np.std(x_train, axis=0)
    return np.divide((x_train - x_mean), x_std), np.divide((x_test - x_mean), x_std)