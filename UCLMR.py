# %%
import pandas as pd
from fnc.refs.utils.dataset import DataSet as ds
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import make_scorer,f1_score
import tensorflow as tf
from tensorflow.keras import activations
from keras import regularizers
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, validation_curve, StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor
import numpy as np
from numpy.random import seed
# import shap
import pydotplus
import graphviz
import warnings
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA

from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig


from fnc.refs.utils.dataset import DataSet
from fnc.refs.utils.generate_test_splits import kfold_split, get_stances_for_folds
from fnc.refs.feature_engineering import gen_non_bleeding_feats, gen_or_load_feats
from fnc.utils.utilities import *
from fnc.utils.score_calculation import *
from fnc.utils import printout_manager
# %matplotlib inline
# shap.initjs()

# %%
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# %%
#dataset pre-processing
train_stances_set = pd.read_csv("data/fnc-1/datasets/combined_train_stances.csv")
test_stances_set = pd.read_csv("data/fnc-1/datasets/combined_test_stances.csv")
train_bodies_set = pd.read_csv("data/fnc-1/datasets/combined_train_bodies.csv")
test_bodies_set = pd.read_csv("data/fnc-1/datasets/combined_test_bodies.csv")
#train_stances_set = pd.read_csv("data/train_stances.csv")
#test_stances_set = pd.read_csv("data/competition_test_stances.csv")
#train_bodies_set = pd.read_csv("data/train_bodies.csv")
#test_bodies_set = pd.read_csv("data/competiotion_test_bodies.csv")
train= train_stances_set.set_index('Body ID').join(train_bodies_set.set_index('Body ID'),lsuffix='_caller', rsuffix='_other')
test= test_stances_set.set_index('Body ID').join(test_bodies_set.set_index('Body ID'),lsuffix='_caller', rsuffix='_other')
train = train.reset_index()
test = test.reset_index()
train_heads= train['Headline'].unique().tolist()
test_heads= test['Headline'].unique().tolist()
train_bodies = train['articleBody'].unique().tolist()
test_bodies = test['articleBody'].unique().tolist()

# %%
stop_words = [
        "a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along",
        "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
        "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be",
        "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
        "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "co",
        "con", "could", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight",
        "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
        "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first", "five", "for",
        "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had",
        "has", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
        "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed", "interest",
        "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made",
        "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much",
        "must", "my", "myself", "name", "namely", "neither", "nevertheless", "next", "nine", "nobody", "now", "nowhere",
        "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours",
        "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see",
        "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some",
        "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take",
        "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
        "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though",
        "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve",
        "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what",
        "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon",
        "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will",
        "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
        ]

# %%
lim_unigram = 5000

# Create vectorizers and BOW and TF arrays for train set
bow_vectorizer = CountVectorizer(max_features=lim_unigram, stop_words=stop_words)
bow = bow_vectorizer.fit_transform(train_heads + train_bodies)  # Train set only

tfreq_vectorizer = TfidfTransformer(use_idf=False).fit(bow)
tfreq = tfreq_vectorizer.transform(bow).toarray()  # Train set only

# %%
train['tf_headline_index'] = 0
train['tf_body_index'] = 0
for i in range(0,len(train_heads)):
    train.loc[train['Headline'] == train_heads[i], 'tf_headline_index'] = i
for i in range(len(train_heads),len(tfreq)):
    train.loc[train['articleBody'] == train_bodies[i-len(train_heads)], 'tf_body_index'] = i

# %%
train.loc[train['Stance'] == 'unrelated', 'Stance'] = 3
train.loc[train['Stance'] == 'discuss', 'Stance'] = 2
train.loc[train['Stance'] == 'agree', 'Stance'] = 0
train.loc[train['Stance'] == 'disagree', 'Stance'] = 1

# %%
test.loc[test['Stance'] == 'unrelated', 'Stance'] = 3
test.loc[test['Stance'] == 'discuss', 'Stance'] = 2
test.loc[test['Stance'] == 'agree', 'Stance'] = 0
test.loc[test['Stance'] == 'disagree', 'Stance'] = 1

# %%
tfidf_vectorizer = TfidfVectorizer(max_features=lim_unigram, stop_words=stop_words).\
fit(train_heads + train_bodies + test_heads + test_bodies)  # Train and test sets

# %%
X_train = []
for i in range (0, len(train)):
    head_tf=tfreq[train.loc[i]['tf_headline_index']].reshape(1, -1)
    body_tf=tfreq[train.loc[i]['tf_body_index']].reshape(1, -1)
    head_tfidf = tfidf_vectorizer.transform([train.loc[i]['Headline']]).toarray()
    body_tfidf = tfidf_vectorizer.transform([train.loc[i]['articleBody']]).toarray()
    tfidf_cos = cosine_similarity(head_tfidf, body_tfidf)[0].reshape(1, 1)
    feat_vec = np.squeeze(np.c_[head_tf, body_tf, tfidf_cos])
    X_train.append(feat_vec)

# %%
X_test = []

for index, instance in test.iterrows() :
    head = instance['Headline']
    body = instance['articleBody']
    head_bow = bow_vectorizer.transform([head]).toarray()
    head_tf = tfreq_vectorizer.transform(head_bow).toarray()[0].reshape(1, -1)
    head_tfidf = tfidf_vectorizer.transform([head]).toarray().reshape(1, -1)
    body_bow = bow_vectorizer.transform([body]).toarray()
    body_tf = tfreq_vectorizer.transform(body_bow).toarray()[0].reshape(1, -1)
    body_tfidf = tfidf_vectorizer.transform([body]).toarray().reshape(1, -1)
    tfidf_cos = cosine_similarity(head_tfidf, body_tfidf)[0].reshape(1, 1)
    feat_vec = np.squeeze(np.c_[head_tf, body_tf, tfidf_cos])
    X_test.append(feat_vec)

# %%
model = Sequential()
model.add(Dense(100, activation='relu', kernel_regularizer=regularizers.l2(l2=0.0001),bias_regularizer =regularizers.l2(l2=0.0001),activity_regularizer = regularizers.l2(l2=0.0001),input_dim=10001))
model.add(Dropout(rate=0.4, seed=27))
model.add(Dense(4, kernel_regularizer=regularizers.l2(l2=0.0001),bias_regularizer =regularizers.l2(l2=0.0001),activity_regularizer = regularizers.l2(l2=0.0001)))
model.add(Dropout(rate=0.4, seed=27))
model.add(tf.keras.layers.Softmax())
model.compile(loss='SparseCategoricalCrossentropy', optimizer= tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['SparseCategoricalCrossentropy'])

# %%
y_train = np.asarray(train['Stance']).astype('int32')
X_train = np.asarray(X_train).astype('float32')
y_test = np.asarray(test['Stance']).astype('int32')
X_test = np.asarray(X_test).astype('float32')

# %%
model.fit(X_train,y_train,batch_size=500,epochs=90)

# %%
y_pred = model.predict(X_train)
train_pred = []
for a in y_pred:
    train_pred.append(a.argmax())

# %%
errors = train_pred - y_train
print(f"Accuracy: {len(errors[errors == 0]) / len(errors)}")

# %%
conf_mat = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
for i in range(0,len(train_pred)):
    conf_mat[y_train[i]][train_pred[i]] = conf_mat[y_train[i]][train_pred[i]] + 1
conf_mat

# %%
for i in range(0,4) :
    print(f"Accuracy class {i} : {conf_mat[i][i] / sum(conf_mat[i])}")

get_accuracy(train_pred, y_train, stance=True)
get_f1score(train_pred, y_train, stance=True)
report_score(y_train, train_pred)

# %%
y_pred = model.predict(X_test)
test_pred = []
for a in y_pred:
    test_pred.append(a.argmax())
errors = test_pred - y_test
print(f"Accuracy: {len(errors[errors == 0]) / len(errors)}")

# %%
conf_mat = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
for i in range(0,len(test_pred)):
    conf_mat[y_test[i]][test_pred[i]] = conf_mat[y_test[i]][test_pred[i]] + 1
conf_mat 

# %%
for i in range(0,4) :
    print(f"Accuracy class {i} : {conf_mat[i][i] / sum(conf_mat[i])}")

get_accuracy(test_pred, y_test, stance=True)
get_f1score(test_pred, y_test, stance=True)
report_score(y_test, test_pred)

# %%



