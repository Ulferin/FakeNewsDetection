from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import numpy as np

import torch

from utils import *

debug = False
debug_models = ''

model_name = 'roberta'
model_type = "roberta-base"

train_ep = 5
learning_rate = 1e-5
pol_dec_end = 1e-7
pol_dec_pow = 1.0
max_seq_len = 512
batch_size = 4
proc_count = 8
do_lower_case = False

saved = False
save_folder = f'./models/roberta_base_confronto_Attardi'

LABELS = {
    'agree': 0,
    'disagree': 1,
    'discuss': 2,
    'unrelated': 3
}

#dataset pre-processing
train_stances_set = pd.read_csv("data_simple/combined_train_stances.csv")
test_stances_set = pd.read_csv("data_simple/combined_test_stances.csv")
train_bodies_set = pd.read_csv("data_simple/combined_train_bodies.csv")
test_bodies_set = pd.read_csv("data_simple/combined_test_bodies.csv")

train = train_stances_set.set_index('Body ID').join(train_bodies_set.set_index('Body ID'),lsuffix='_caller', rsuffix='_other')
test = test_stances_set.set_index('Body ID').join(test_bodies_set.set_index('Body ID'),lsuffix='_caller', rsuffix='_other')
train = train.reset_index()
test = test.reset_index()

train = train.replace({'Stance': LABELS})
train.pop('Body ID')
train.columns = ["text_a", "labels", "text_b"]
train = train[['text_a', 'text_b', 'labels']]
y_train = train['labels']

test = test.replace({'Stance': LABELS})
test.pop('Body ID')
y_test = test.pop('Stance')
test.columns = ["text_a", 'text_b']


X_test = test.values.tolist()

X_train = train

if not debug:
    if not saved:
        # Optional model configuration
        model_args = ClassificationArgs(num_train_epochs=train_ep,
                                        learning_rate = learning_rate,
                                        polynomial_decay_schedule_lr_end = pol_dec_end,
                                        polynomial_decay_schedule_power = pol_dec_pow,
                                        dataloader_num_workers = 2,
                                        do_lower_case = do_lower_case,
                                        manual_seed = 42,
                                        reprocess_input_data = True,
                                        overwrite_output_dir = True,
                                        process_count = proc_count,
                                        train_batch_size = batch_size,
                                        max_seq_length = max_seq_len,
                                        fp16 = True,
                                        no_save = False,
                                        save_eval_checkpoints = False,
                                        save_model_every_epoch = True,
                                        save_optimizer_and_scheduler = False,
                                        # sliding_window = True, # only if window is to be used
                                        # stride = 0.8, # stride * max_seq_len step between windows
                                        # tie_value = 3
                                    )

        # Create a ClassificationModel
        model = ClassificationModel(model_name,
                                    model_type,
                                    num_labels = 4,
                                    use_cuda = torch.cuda.is_available(),
                                    cuda_device = 0,
                                    args=model_args
                                    )

        # Train the model
        model.train_model(X_train)
        torch.save(model, save_folder)
    else:
        print(f"Loading model saved in: {save_folder}")
        model = torch.load(save_folder)

    # Make predictions with the model
    predictions, raw_outputs = model.predict(X_test)


    conf_mat = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    for i in range(0,len(predictions)):
        conf_mat[y_test[i]][predictions[i]] = conf_mat[y_test[i]][predictions[i]] + 1

    confusion_matrix = np.array(conf_mat)

    def process_cm(confusion_mat, i=0, to_print=True):
        # i means which class to choose to do one-vs-the-rest calculation
        # rows are actual obs whereas columns are predictions
        tp = confusion_mat[i,i]  # correctly labeled as i
        fp = confusion_mat[:,i].sum() - tp  # incorrectly labeled as i
        fn = confusion_mat[i,:].sum() - tp  # incorrectly labeled as non-i
        tn = confusion_mat.sum().sum() - tp - fp - fn
        if to_print:
            print('TP: {}'.format(tp))
            print('FP: {}'.format(fp))
            print('FN: {}'.format(fn))
            print('TN: {}'.format(tn))

        prec = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2*(prec*recall)/(prec+recall)

        print(f"Accuracy class {i} : {(tp+tn)/(tp+tn+fp+fn)}")
        print(f"f1 score class {i} : {f1}\n")

    for i in range(4):
        print('Calculating 2x2 contigency table for label{}'.format(i))
        process_cm(confusion_matrix, i, to_print=False)

    report_score(y_test, predictions)
    print()

    def compute_fp(cm, i):
        fp = 0
        for j in range(len(cm)):
            fp += cm[j][i]

        return fp-cm[i][i]


    def compute_tn(cm, i):
        total = 0
        for j in range(len(cm)):
            total = total + sum(cm[j]) - cm[j][i]

        return total - sum(cm[i]) + cm[i][i]

    for i in range(0,4) :
        tp = conf_mat[i][i]
        fp = compute_fp(conf_mat, i)
        fn = sum(conf_mat[i]) - conf_mat[i][i]
        tn = compute_tn(conf_mat, i)

        prec = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2*(prec*recall)/(prec+recall)

        print(f"Accuracy class {i} : {(tp+tn)/(tp+tn+fp+fn)}")
        print(f"f1 score class {i} : {f1}\n")

    print()
    print("----- Model accuracy -----")
    print("Per stance accuracy: ", get_accuracy(predictions, y_test, stance=True))
    print("Related/Unrelated accuracy: ", get_accuracy(predictions, y_test, stance=False))

    print("----- Model f1 -----")
    print("Per stance f1 macro: ", get_f1score(predictions, y_test, stance=True))
    print("Related/Unrelated f1 macro: ", get_f1score(predictions, y_test, stance=False))