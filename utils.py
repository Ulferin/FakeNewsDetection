from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
LABELS_dict = {
    'agree': 0,
    'disagree': 1,
    'discuss': 2,
    'unrelated': 3
}

RELATED = LABELS[0:3]

def load_datasets():
    """Loads and preprocess the combined traininig and test datasets.
    Returns the polished datasets and labels. 
    """

    train_stances_set = pd.read_csv("data_simple/combined_train_stances.csv")
    test_stances_set = pd.read_csv("data_simple/combined_test_stances.csv")
    train_bodies_set = pd.read_csv("data_simple/combined_train_bodies.csv")
    test_bodies_set = pd.read_csv("data_simple/combined_test_bodies.csv")

    train = train_stances_set.set_index('Body ID').join(train_bodies_set.set_index('Body ID'),lsuffix='_caller', rsuffix='_other')
    test = test_stances_set.set_index('Body ID').join(test_bodies_set.set_index('Body ID'),lsuffix='_caller', rsuffix='_other')
    train = train.reset_index()
    test = test.reset_index()

    train = train.replace({'Stance': LABELS_dict})
    train.pop('Body ID')
    train.columns = ["text_a", "labels", "text_b"]
    train = train[['text_a', 'text_b', 'labels']]
    y_train = train['labels']

    test = test.replace({'Stance': LABELS_dict})
    test.pop('Body ID')
    y_test = test.pop('Stance')
    test.columns = ["text_a", 'text_b']


    X_test = test.values.tolist()

    X_train = train
    return X_train, y_train, X_test, y_test


def get_accuracy(y_predicted, y_true, stance=False):
    """Computes accuracy from the predicted and true labels.

    Parameters
    ----------
    y_predicted :
        Predicted labels
    y_true : 
        Ground truth
    stance : optional
        If True, uses the stance-related setting, otherwise uses the
        stance-agnostic setting, by default False

    Returns
    -------
        Accuracy score for the specified setting.
    """

    y_true_temp = []
    y_predicted_temp = []
    if stance == False:
        for y in y_true:
            if y >= 0 and y <= 2: #'agree', 'disagree', 'discuss'
                y_true_temp.append(1)   # related
            else:
                y_true_temp.append(0) # unrelated

        for y in y_predicted:
            if y >= 0 and y <= 2: #'agree', 'disagree', 'discuss'
                y_predicted_temp.append(1)   # related
            else:
                y_predicted_temp.append(0) # unrelated

        return accuracy_score(y_true_temp, y_predicted_temp)

    else:
        return accuracy_score(y_true, y_predicted)


def get_f1score(y_predicted, y_true, stance=False):
    """Computes f1 macro score from the predicted and true labels.

    Parameters
    ----------
    y_predicted :
        Predicted labels
    y_true : 
        Ground truth
    stance : optional
        If True, uses the stance-related setting, otherwise uses the
        stance-agnostic setting, by default False

    Returns
    -------
        F1 macro score for the specified setting.
    """

    y_true_temp = []
    y_predicted_temp = []
    if stance == False:
        for y in y_true:
            if y >= 0 and y <= 2: #'agree', 'disagree', 'discuss'
                y_true_temp.append(1)   # related
            else:
                y_true_temp.append(0) # unrelated

        for y in y_predicted:
            if y >= 0 and y <= 2: #'agree', 'disagree', 'discuss'
                y_predicted_temp.append(1)   # related
            else:
                y_predicted_temp.append(0) # unrelated

        return f1_score(y_true_temp, y_predicted_temp, average='macro')

    else:
        return f1_score(y_true, y_predicted, average='macro')


def confusion_matrix_string(cm):
    """Transform the passed confusion matrix into a printable string.

    Parameters
    ----------
    cm :
        Confusion matrix as numpy array.

    Returns
    -------
        Returns confusion matrix as string.
    """
    
    lines = []
    header = "|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format('', *LABELS)
    line_len = len(header)
    lines.append("-"*line_len)
    lines.append(header)
    lines.append("-"*line_len)

    hit = 0
    total = 0
    for i, row in enumerate(cm):
        hit += row[i]
        total += sum(row)
        lines.append("|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format(LABELS[i],
                                                                   *row))
        lines.append("-"*line_len)
    return '\n'.join(lines)


def compute_scores(confusion_mat, i):
    """Computes f1 and accuracy for the specified class using confusion matrix.

    Parameters
    ----------
    confusion_mat :
        Confusion matrix
    i : 
        Class to consider to do one-vs-the-rest calculation.

    Returns
    -------
        Returns accuracy and f1 score for the given class.
    """

    tp = confusion_mat[i,i] + 1e-16
    fp = confusion_mat[:,i].sum() - tp + 1e-16
    fn = confusion_mat[i,:].sum() - tp + 1e-16
    tn = confusion_mat.sum().sum() - tp - fp - fn + 1e-16

    prec = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*(prec*recall)/(prec+recall)

    acc = f"Accuracy class {i} : {(tp+tn)/(tp+tn+fp+fn)}\n"
    f1_s = f"f1 score class {i} : {f1}\n\n"

    return acc, f1_s


def save_stats(y_test, predictions, configuration):
    """Prints the computed stats to a text file in ./results.

    Parameters
    ----------
    y_test :
        Ground truth
    predictions :
        Predicted labels
    configuration : 
        Configuration dictionary for the used model
    """

    conf_mat = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    for i in range(0,len(predictions)):
        conf_mat[y_test[i]][predictions[i]] += 1

    confusion_matrix = np.array(conf_mat)

    cm_s = confusion_matrix_string(confusion_matrix)

    with open(f"./results/{configuration['model_name']}.txt", 'a') as f:
        f.write(f"Configuration:\n")
        f.write(f"{configuration}\n\n")

        f.write(f"{cm_s}\n")

        for i in range(4):
            acc, f1_s = compute_scores(confusion_matrix, i)
            f.write(acc)
            f.write(f1_s)

        f.write("\n")
        f.write("----- Model accuracy -----\n")
        f.write(f"Per stance accuracy: {get_accuracy(predictions, y_test, stance=True)}\n")
        f.write(f"Related/Unrelated accuracy: {get_accuracy(predictions, y_test, stance=False)}\n")

        f.write("----- Model f1 -----\n")
        f.write(f"Per stance f1 macro: {get_f1score(predictions, y_test, stance=True)}\n")
        f.write(f"Related/Unrelated f1 macro: {get_f1score(predictions, y_test, stance=False)}\n\n\n")