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
    #dataset pre-processing
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
    # if stance == False convert into 2-class-problem
    y_true_temp = [] # don't use parameters since it will change them by reference
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
    # if stance == False convert into 2-class-problem
    y_true_temp = [] # don't use parameters since it will change them by reference
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


def score_submission(gold_labels, test_labels):
    score = 0.0
    cm = [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]

    for i, (g, t) in enumerate(zip(gold_labels, test_labels)):
        g_stance, t_stance = g, t
        if g_stance == t_stance:
            score += 0.25
            if g_stance != LABELS.index('unrelated'):
                score += 0.50
        if g_stance in RELATED and t_stance in RELATED:
            score += 0.25

        cm[g_stance][t_stance] += 1

    return score, cm


def print_confusion_matrix(cm):
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


def report_score(actual,predicted):
    score,cm = score_submission(actual,predicted)
    best_score, _ = score_submission(actual,actual)

    cm_s = print_confusion_matrix(cm)
    print("Score: " +str(score) + " out of " + str(best_score) + "\t("+str(score*100/best_score) + "%)")
    return cm_s, score*100/best_score


def process_cm(confusion_mat, i=0, print_stats=True):
        # i means which class to choose to do one-vs-the-rest calculation
        # rows are actual obs whereas columns are predictions
        tp = confusion_mat[i,i]  # correctly labeled as i
        fp = confusion_mat[:,i].sum() - tp  # incorrectly labeled as i
        fn = confusion_mat[i,:].sum() - tp  # incorrectly labeled as non-i
        tn = confusion_mat.sum().sum() - tp - fp - fn
        if print_stats:
            print('TP: {}'.format(tp))
            print('FP: {}'.format(fp))
            print('FN: {}'.format(fn))
            print('TN: {}'.format(tn))

        prec = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2*(prec*recall)/(prec+recall)

        acc = f"Accuracy class {i} : {(tp+tn)/(tp+tn+fp+fn)}\n"
        f1_s = f"f1 score class {i} : {f1}\n\n"

        return acc, f1_s


def save_stats(y_test, predictions, configuration):
    conf_mat = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    for i in range(0,len(predictions)):
        conf_mat[y_test[i]][predictions[i]] += 1

    confusion_matrix = np.array(conf_mat)

    cm_s, score = report_score(y_test, predictions)

    with open(f"./results/{configuration['model_name']}.txt", 'a') as f:
        f.write(f"Configuration:\n")
        f.write(f"{configuration}\n\n")

        f.write(cm_s)
        f.write(f"\nScore: {score}\n\n")

        for i in range(4):
            acc, f1_s = process_cm(confusion_matrix, i, print_stats=False)
            f.write(acc)
            f.write(f1_s)

        f.write("\n")
        f.write("----- Model accuracy -----\n")
        f.write(f"Per stance accuracy: {get_accuracy(predictions, y_test, stance=True)}\n")
        f.write(f"Related/Unrelated accuracy: {get_accuracy(predictions, y_test, stance=False)}\n")

        f.write("----- Model f1 -----\n")
        f.write(f"Per stance f1 macro: {get_f1score(predictions, y_test, stance=True)}\n")
        f.write(f"Related/Unrelated f1 macro: {get_f1score(predictions, y_test, stance=False)}\n\n\n")