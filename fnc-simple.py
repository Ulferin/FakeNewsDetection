from simpletransformers.classification import ClassificationModel, ClassificationArgs
import numpy as np

import torch

from utils import *


model = 'roberta'  # Configuration to be executed
saved = True   # Set to True to recover an existing model


# Best configurations for each of the tested models
configurations = {

    'bert': {
        'model_name': 'bert',
        'model_type': 'bert-base-uncased',
        'train_ep': 5,
        'learning_rate': 3e-5,
        'max_seq_len': 512,
        'batch_size': 8,
        'scheduler': 'linear_schedule_with_warmup',
        'do_lower_case': True
    },
    'roberta': {
        'model_name': 'roberta',
        'model_type': 'roberta-base',
        'train_ep': 5,
        'learning_rate': 1e-5,
        'pol_dec_end': 1e-7,
        'pol_dec_pow': 1.2,
        'max_seq_len': 512,
        'batch_size': 4,
        'scheduler': 'polynomial_decay_schedule_with_warmup',
        'do_lower_case': False
    },
    'longformer': {
        'model_name': 'longformer',
        'model_type': 'allenai/longformer-base-4096',
        'train_ep': 5,
        'learning_rate': 1e-5,
        'max_seq_len': 512,
        'batch_size': 4,
        'scheduler': 'linear_schedule_with_warmup',
        'do_lower_case': False
    },
    'distilbert': {
        'model_name': 'distilbert',
        'model_type': 'distilbert-base-uncased',
        'train_ep': 5,
        'learning_rate': 3e-5,
        'max_seq_len': 512,
        'batch_size': 16,
        'scheduler': 'linear_schedule_with_warmup',
        'do_lower_case': True
    },
    'deberta': {
        'model_name': 'deberta',
        'model_type': 'microsoft/deberta-base',
        'train_ep': 5,
        'learning_rate': 5e-6,
        'pol_dec_end': 1e-7,
        'pol_dec_pow': 1.0,
        'max_seq_len': 512,
        'batch_size': 2,
        'scheduler': 'polynomial_decay_schedule_with_warmup',
        'do_lower_case': False
    },
    
}

X_train, y_train, X_test, y_test = load_datasets()

# Test the specified model and saves results in ./results/model_name.txt
def train_transformer(configuration):
    model_name = configuration['model_name']
    model_type = configuration['model_type']
    train_ep = configuration['train_ep']
    learning_rate = configuration['learning_rate']
    max_seq_len = configuration['max_seq_len']
    batch_size = configuration['batch_size']
    scheduler = configuration['scheduler']
    do_lower_case = configuration['do_lower_case']

    if scheduler == 'polynomial_decay_schedule_with_warmup':
        pol_dec_end = configuration['pol_dec_end']
        pol_dec_pow = configuration['pol_dec_pow']
    else:
        pol_dec_end = 1e-7
        pol_dec_pow = 1.0

    save_folder = f'./models/{model_name}'

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
                                    train_batch_size = batch_size,
                                    max_seq_length = max_seq_len,
                                    fp16 = True,
                                    no_save = True,
                                    save_eval_checkpoints = False,
                                    save_model_every_epoch = False,
                                    save_optimizer_and_scheduler = False,
                                    scheduler=scheduler
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

    # Make predictions with the model
    predictions, _ = model.predict(X_test)


    save_stats(y_test, predictions, configuration)


train_transformer(configurations[model])
