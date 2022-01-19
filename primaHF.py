# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from sklearn.metrics import accuracy_score,f1_score
import tensorflow as tf
import torch
import gc
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import ElectraModel, ElectraConfig, ElectraTokenizer, ElectraForSequenceClassification, TFAutoModelForSequenceClassification
from datasets import load_dataset, Dataset

# %%
checkpoint = "google/electra-small-discriminator"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
print(tokenizer.model_max_length)

batch_size = 16
num_epochs= 1


# %%
#dataset pre-processing
train_stances_set = pd.read_csv("data/fnc-1/datasets/combined_train_stances.csv")
test_stances_set = pd.read_csv("data/fnc-1/datasets/combined_test_stances.csv")
train_bodies_set = pd.read_csv("data/fnc-1/datasets/combined_train_bodies.csv")
test_bodies_set = pd.read_csv("data/fnc-1/datasets/combined_test_bodies.csv")
#train_stances_set = pd.read_csv("data/train_stances.csv")
#test_stances_set = pd.read_csv("data/competition_test_stances.csv")
#train_bodies_set = pd.read_csv("data/train_bodies.csv")
#test_bodies_set = pd.read_csv("data/competition_test_bodies.csv")
train= train_stances_set.set_index('Body ID').join(train_bodies_set.set_index('Body ID'),lsuffix='_caller', rsuffix='_other')
test= test_stances_set.set_index('Body ID').join(test_bodies_set.set_index('Body ID'),lsuffix='_caller', rsuffix='_other')
train = train.reset_index()
test = test.reset_index()
train.loc[train['Stance'] == 'unrelated', 'Stance'] = 3
train.loc[train['Stance'] == 'discuss', 'Stance'] = 2
train.loc[train['Stance'] == 'agree', 'Stance'] = 0
train.loc[train['Stance'] == 'disagree', 'Stance'] = 1
test.loc[test['Stance'] == 'unrelated', 'Stance'] = 3
test.loc[test['Stance'] == 'discuss', 'Stance'] = 2
test.loc[test['Stance'] == 'agree', 'Stance'] = 0
test.loc[test['Stance'] == 'disagree', 'Stance'] = 1
train.pop('Body ID')
y_train = train.pop('Stance')
train['Stance'] = y_train
train.columns = ["text_a", "text_b", "labels"]
test.pop('Body ID')
y_test = test.pop('Stance')
test.columns = ["text_a", "text_b"]
X_test = test.values.tolist()

# %%
X_train,X_val = train_test_split(train,stratify=train['labels'],test_size=0.30, random_state=42)

# %%
X_train = Dataset.from_pandas(X_train)
X_val = Dataset.from_pandas(X_val)

# %%
def tokenize_function(example):
    return tokenizer(example["text_a"], example["text_b"], truncation=True, max_length=tokenizer.model_max_length)


X_train = X_train.map(tokenize_function, batched=True)
X_val = X_val.map(tokenize_function, batched=True)

# %%
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

# %%
X_train = X_train.to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
)

X_val = X_val.to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
)
# %%
from tensorflow.keras.optimizers.schedules import PolynomialDecay

# The number of training steps is the number of samples in the dataset, divided by the batch size then multiplied
# by the total number of epochs
num_train_steps = len(X_train) * num_epochs
lr_scheduler = PolynomialDecay(
    initial_learning_rate=0.01, end_learning_rate=0.0, decay_steps=num_train_steps
)
from tensorflow.keras.optimizers import Adam

opt = Adam(learning_rate=4e-5)

# %%
import tensorflow as tf
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=4)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

# %%
model.fit(X_train,validation_data=X_val, epochs=num_epochs)
model.save_pretrained("./models")
# %%



