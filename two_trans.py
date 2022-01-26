# %%
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import numpy as np

import torch

from utils import *

# %%
ckp = False

model_name = 'roberta'
model_type = "roberta-large"

train_ep = 5
learning_rate = 1e-5
pol_dec_end = 1e-7
pol_dec_pow = 1.0
max_seq_len = 512
batch_size = 4
proc_count = 8
do_lower_case = False

saved = True
save_folder = f'./models/roberta_base_6batch_higherLR'


X_train, y_train, X_test, y_test = load_datasets()

# %%
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
                                save_optimizer_and_scheduler = False
            )
if not saved:
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

# %%
# Make predictions with the model
predictions, raw_outputs = model.predict(X_test)


# %%
conf_mat = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
for i in range(0,len(predictions)):
    conf_mat[y_test[i]][predictions[i]] += 1

confusion_matrix = np.array(conf_mat)

report_score(y_test, predictions)
print()

for i in range(4):
    process_cm(confusion_matrix, i, print_stats=False)

print()
print("----- Model accuracy -----")
print("Per stance accuracy: ", get_accuracy(predictions, y_test, stance=True))
print("Related/Unrelated accuracy: ", get_accuracy(predictions, y_test, stance=False))

print("----- Model f1 -----")
print("Per stance f1 macro: ", get_f1score(predictions, y_test, stance=True))
print("Related/Unrelated f1 macro: ", get_f1score(predictions, y_test, stance=False))

from sklearn.metrics import f1_score

def calculate_f1_scores(y_true, y_predicted):
    f1_macro = f1_score(y_true, y_predicted, average='macro')
    f1_classwise = f1_score(y_true, y_predicted, average=None, labels=[0, 1, 2, 3])

    resultstring = "F1 macro: {:.3f}".format(f1_macro * 100) + "% \n"
    resultstring += "F1 agree: {:.3f}".format(f1_classwise[0] * 100) + "% \n"
    resultstring += "F1 disagree: {:.3f}".format(f1_classwise[1] * 100) + "% \n"
    resultstring += "F1 discuss: {:.3f}".format(f1_classwise[2] * 100) + "% \n"
    resultstring += "F1 unrelated: {:.3f}".format(f1_classwise[3] * 100) + "% \n"

    return resultstring

print(calculate_f1_scores(predictions, y_test))


# %%
preds = np.array(predictions, dtype=int)
idx_rel = np.where(preds != 3)
idx_unrel = np.where(preds == 3)
idx_rel = list(idx_rel[0])

# %%
X_test_df = pd.DataFrame(X_test, columns =['text_a', 'text_b'])

# %%
X_test_rel = X_test_df.iloc[idx_rel]
# y_test_rel = y_test.iloc[idx]

X_test_rel.shape

# %%
X_test_rel

# %%
X_train_rel = X_train.loc[X_train['labels'] != 3]
X_train_rel

# %%
model_name = 'roberta'
model_type = "cardiffnlp/twitter-roberta-base-sentiment"

saved = False
save_folder = f'./models/roberta_sentiment'

train_ep = 5
learning_rate = 1e-5
pol_dec_end = 1e-7
pol_dec_pow = 1.0
max_seq_len = 512
batch_size = 4
proc_count = 8
do_lower_case = False

LABELS_sent = {
    0:2,
    1:0,
    2:1    
}
X_train_rel = X_train_rel.replace({'labels': LABELS_sent})
X_train_rel


# %%
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
                                save_optimizer_and_scheduler = False
            )
if not saved:
    model = ClassificationModel(model_name,
                                model_type,
                                num_labels = 3,
                                use_cuda = torch.cuda.is_available(),
                                cuda_device = 0,
                                args=model_args
            )

    # Train the model
    model.train_model(X_train_rel)
    torch.save(model, save_folder)


