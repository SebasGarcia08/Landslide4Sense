import numpy as np
import typing as ty


def eval_image(
    predict: np.ndarray, label: np.ndarray, num_classes: int
) -> ty.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    index = np.where((label >= 0) & (label < num_classes))
    predict = predict[index]
    label = label[index]

    TP = np.zeros((num_classes, 1))
    FP = np.zeros((num_classes, 1))
    TN = np.zeros((num_classes, 1))
    FN = np.zeros((num_classes, 1))

    for i in range(0, num_classes):
        TP[i] = np.sum(label[np.where(predict == i)] == i)
        FP[i] = np.sum(label[np.where(predict == i)] != i)
        TN[i] = np.sum(label[np.where(predict != i)] != i)
        FN[i] = np.sum(label[np.where(predict != i)] == i)

    return TP, FP, TN, FN, len(label)


def import_name(module_name: str, name: str):
    """Import a named object from a module in the context of this function."""
    module = __import__(module_name, globals(), locals(), [name])
    return vars(module)[name]
