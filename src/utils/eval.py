import numpy as np
import os
from sklearn.metrics import confusion_matrix

def eval_labels(labels, pred, save_path="../tmp/calc_log", filename='confusion_matrix.csv'):
    """Evaluate performance of predicted labels
    Args:
        labels: list of true labels
        pred: list of predicted labels
        save_path: String path to a folder for saving confusion matrix in csv
        filename: String name of file name
    Returns:
    """
    if not os.path.isdir(save_path):
            os.makedirs(save_path)
    conf_matrix = confusion_matrix(labels, pred)
    print(conf_matrix)
    np.savetxt(os.path.join(save_path, filename), conf_matrix, delimiter=',')
    
    # Create a boolean array whether each image is correctly classified.
    correct = (labels == pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()
    acc = float(correct_sum) /len(pred)
    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, len(pred)))
    return