import numpy as np


def prob_to_acc(
    prob: np.ndarray, actual_labels: np.ndarray, class_label: np.ndarray
) -> float:
    """
    Calculate the accuracy of a classification model with multiple target classes.
    """
    ix_max = np.argmax(prob, axis=1)
    predicted_labels = np.asarray([class_label[ix] for ix in ix_max])
    return (actual_labels == predicted_labels).sum() / len(actual_labels)
