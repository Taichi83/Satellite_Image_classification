import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import os
import numpy as np

## visualize input data
def barplot_classes(labels_oh, cl_nms, save_path="../tmp/figures", filename='class_dist.png'):
    """Save bar plot of each class volumes
    Args:
        labels_oh: Binary array in Numpy array format [-1, number of classes]
        cl_nms: Numpy array of strings; class names [number of classes]
        save_path: String path to a folder for saving bar plot image
        filename: String name of file name
    Returns:
    """
    left = [cl_nms[i] for i in range(len(cl_nms))]
    height = [labels_oh.sum(axis=0)[i] for i in range(len(cl_nms))]
    plt.bar(left, height)
    plt.xticks(rotation=90)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, filename))
    return

## visualize sample images
def sample_images(images, labels_oh, cl_nms, save_path="../tmp/figures", filename='sample_images.png'):
    """Save sample images with their labels name
    Args:
        images: Numpy array of images [-1, hight, width, dims]
        labels_oh: Binary array in Numpy array format [-1, number of classes]
        cl_nms: Numpy array of strings; class names [number of classes]
        save_path: String path to a folder for saving bar plot image
        filename: String name of file name
    Returns:
    """
    new_style = {'grid': False}
    plt.rc('axes', **new_style)
    _, ax = plt.subplots(2, 8, sharex='col', sharey='row', figsize=(20, 5.5), facecolor='w')
    i = 0
    for i in range(16):
        j = np.random.randint(labels_oh.shape[0])
        ax[i // 8, i % 8].imshow(images[j])
        ax[i // 8, i % 8].set_title('train #:%d \n class:%s' %(j, cl_nms[labels_oh[j]==1][0]))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, filename))
    return