from simpletransformers.classification import ClassificationModel, ClassificationArgs
import numpy as np

import torch

from utils import *

ckp = False

# model_name = 'bigbird'
# model_type = "google/bigbird-roberta-base"
model_name = 'roberta'
# model_type = "roberta-base"
model_type = "roberta-large"
# model_name = 'longformer'
# model_type = "allenai/longformer-base-4096"

train_ep = 5
learning_rate = 1e-5
pol_dec_end = 1e-7
pol_dec_pow = 1.0
max_seq_len = 512
batch_size = 4
proc_count = 8
do_lower_case = False

saved = False
save_folder = f'./models/roberta-large_5ep_4bs_1e-5_1e-7'


X_train, y_train, X_test, y_test = load_datasets()

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
if not ckp:
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

else:
    model = ClassificationModel(model_name,
                                'outputs/checkpoint-32104-epoch-2',
                                num_labels=4,
                                use_cuda = torch.cuda.is_available(),
                                cuda_device = 0,
                                args=model_args)
    model.train_model(X_train)

# Make predictions with the model
predictions, raw_outputs = model.predict(X_test)


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
