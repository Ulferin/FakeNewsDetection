from simpletransformers.classification import ClassificationModel, ClassificationArgs
import numpy as np

import torch

from utils import *


# model_name = 'roberta'
# model_type = "roberta-base"
# model_name = 'longformer'
# model_type = "allenai/longformer-base-4096"

proc_count = 8
do_lower_case = False

configurations = [

    # Comparison test 2 batch vs 4 batch
    {
        'model_name': 'roberta',
        'model_type': 'roberta-base',
        'train_ep': 3,
        'learning_rate': 1e-5,
        'pol_dec_end': 1e-7,
        'pol_dec_pow': 1.0,
        'max_seq_len': 512,
        'batch_size': 4,
    },

    # Decreasing learning rate tests
    {
        'model_name': 'roberta',
        'model_type': 'roberta-base',
        'train_ep': 5,
        'learning_rate': 5e-6,
        'pol_dec_end': 1e-7,
        'pol_dec_pow': 1.0,
        'max_seq_len': 512,
        'batch_size': 4,
    },
    {
        'model_name': 'roberta',
        'model_type': 'roberta-base',
        'train_ep': 5,
        'learning_rate': 5e-6,
        'pol_dec_end': 1e-9,
        'pol_dec_pow': 1.0,
        'max_seq_len': 512,
        'batch_size': 4,
    },

    # Higher learning rate tests
    {
        'model_name': 'roberta',
        'model_type': 'roberta-base',
        'train_ep': 5,
        'learning_rate': 5e-5,
        'pol_dec_end': 1e-7,
        'pol_dec_pow': 1.0,
        'max_seq_len': 512,
        'batch_size': 4,
    },
    {
        'model_name': 'roberta',
        'model_type': 'roberta-base',
        'train_ep': 5,
        'learning_rate': 1e-4,
        'pol_dec_end': 1e-7,
        'pol_dec_pow': 1.0,
        'max_seq_len': 512,
        'batch_size': 4,
    },
    
]

ckp = False
saved = False

X_train, y_train, X_test, y_test = load_datasets()

def train_transformer(configuration):
    model_name = configuration['model_name']
    model_type = configuration['model_type']
    train_ep = configuration['train_ep']
    learning_rate = configuration['learning_rate']
    pol_dec_end = configuration['pol_dec_end']
    pol_dec_pow = configuration['pol_dec_pow']
    max_seq_len = configuration['max_seq_len']
    batch_size = configuration['batch_size']

    save_folder = f'./models/{model_type}_{batch_size}batch_{learning_rate}_{pol_dec_end}_{pol_dec_pow}'

    print(f"Running:\n {configuration}\n\nIt will be saved in: {save_folder}\n")

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
                                    no_save = True,
                                    save_eval_checkpoints = False,
                                    save_model_every_epoch = False,
                                    save_optimizer_and_scheduler = False,
                                    scheduler='polynomial_decay_schedule_with_warmup'
                )
    if not ckp:
        # TODO: magari qui cambiare con "se il modello già esiste su disco"
        #       invece che basarsi su parametro
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
    predictions, _ = model.predict(X_test)


    conf_mat = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    for i in range(0,len(predictions)):
        conf_mat[y_test[i]][predictions[i]] += 1

    confusion_matrix = np.array(conf_mat)

    cm_s, score = report_score(y_test, predictions)

    with open(f"./results/{model_type}.txt", 'a') as f:
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

for conf in configurations:
    train_transformer(conf)
