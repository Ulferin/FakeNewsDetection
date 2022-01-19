from sklearn.metrics import accuracy_score, f1_score
from sklearn import feature_extraction

import tensorflow as tf
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

import numpy as np
import pandas as pd

from scipy import spatial
import re
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
_wnl = nltk.WordNetLemmatizer()

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
LABELS_dict = {
    'agree': 0,
    'disagree': 1,
    'discuss': 2,
    'unrelated': 3
}

LABELS_RELATED = ['unrelated','related']
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
    print('\n'.join(lines))


def report_score(actual,predicted):
    score,cm = score_submission(actual,predicted)
    best_score, _ = score_submission(actual,actual)

    print_confusion_matrix(cm)
    print("Score: " +str(score) + " out of " + str(best_score) + "\t("+str(score*100/best_score) + "%)")
    return score*100/best_score


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

        print(f"Accuracy class {i} : {(tp+tn)/(tp+tn+fp+fn)}")
        print(f"f1 score class {i} : {f1}\n")


##########################
# --- UCLMR UTILITIES ---
##########################
def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric

    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()

def normalize_word(w):
    return _wnl.lemmatize(w).lower()

def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]

def word_overlap_features(df):
    X = []
    for i, j in df.iterrows():
        clean_body = clean(j['articleBody'])
        clean_body = get_tokenized_lemmas(clean_body)
        clean_head = clean(j['Headline'])
        clean_head = get_tokenized_lemmas(clean_head)
        features = [
            len(set(clean_head).intersection(clean_body)) / float(len(set(clean_head).union(clean_body)))]
        X.append(features)
    return X


def refuting_features(df):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        # 'refute',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]
    X = []
    for i, j in df.iterrows():
        clean_head = clean(j['Headline'])
        clean_head = get_tokenized_lemmas(clean_head)
        features = [1 if word in clean_head else 0 for word in _refuting_words]
        X.append(features)
    return X


def polarity_features(df):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]

    def calculate_polarity(text):
        tokens = get_tokenized_lemmas(text)
        return sum([t in _refuting_words for t in tokens]) % 2
    X = []
    for i, j in df.iterrows():
        clean_body = clean(j['articleBody'])
        clean_head = clean(j['Headline'])
        features = []
        features.append(calculate_polarity(clean_head))
        features.append(calculate_polarity(clean_body))
        X.append(features)
    return np.array(X)


def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]


def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def chargrams(input, n):
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def append_chargrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in chargrams(" ".join(remove_stopwords(text_headline.split())), size)]
    grams_hits = 0
    grams_early_hits = 0
    grams_first_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
        if gram in text_body[:100]:
            grams_first_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    features.append(grams_first_hits)
    return features


def append_ngrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in ngrams(text_headline, size)]
    grams_hits = 0
    grams_early_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    return features


def hand_features(df):

    def binary_co_occurence(headline, body):
        # Count how many times a token in the title
        # appears in the body text.
        bin_count = 0
        bin_count_early = 0
        for headline_token in clean(headline).split(" "):
            if headline_token in clean(body):
                bin_count += 1
            if headline_token in clean(body)[:255]:
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def binary_co_occurence_stops(headline, body):
        # Count how many times a token in the title
        # appears in the body text. Stopwords in the title
        # are ignored.
        bin_count = 0
        bin_count_early = 0
        for headline_token in remove_stopwords(clean(headline).split(" ")):
            if headline_token in clean(body):
                bin_count += 1
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def count_grams(headline, body):
        # Count how many times an n-gram of the title
        # appears in the entire body, and intro paragraph

        clean_body = clean(body)
        clean_headline = clean(headline)
        features = []
        features = append_chargrams(features, clean_headline, clean_body, 2)
        features = append_chargrams(features, clean_headline, clean_body, 8)
        features = append_chargrams(features, clean_headline, clean_body, 4)
        features = append_chargrams(features, clean_headline, clean_body, 16)
        features = append_ngrams(features, clean_headline, clean_body, 2)
        features = append_ngrams(features, clean_headline, clean_body, 3)
        features = append_ngrams(features, clean_headline, clean_body, 4)
        features = append_ngrams(features, clean_headline, clean_body, 5)
        features = append_ngrams(features, clean_headline, clean_body, 6)
        return features

    X = []
    for i, j in df.iterrows():
        X.append(binary_co_occurence(j['Headline'], j['articleBody'])
                 + binary_co_occurence_stops(j['Headline'], j['articleBody'])
                 + count_grams(j['Headline'], j['articleBody']))


    return X


def CreateModel(neurons,reg,drop, input_dim):
    model = Sequential()
    model.add(Dense(neurons, activation='relu', kernel_regularizer=regularizers.l2(l2=reg),bias_regularizer =regularizers.l2(l2=reg),activity_regularizer = regularizers.l2(l2=reg),input_dim=input_dim))
    model.add(Dropout(rate=drop, seed=27))
    model.add(Dense(4, kernel_regularizer=regularizers.l2(l2=reg),bias_regularizer =regularizers.l2(l2=reg),activity_regularizer = regularizers.l2(l2=reg)))
    model.add(Dropout(rate=drop, seed=27))
    model.add(tf.keras.layers.Softmax())
    model.compile(loss='SparseCategoricalCrossentropy', optimizer= tf.keras.optimizers.Adam(learning_rate=0.01))
    return model


def doc_2_vec_features(df, vec_size = 3000, win_size = 5):
    docs = []
    for i, j in df.iterrows():
        clean_body = clean(j['articleBody'])
        clean_body = get_tokenized_lemmas(clean_body)
        docs.append(clean_body)
        clean_head = clean(j['Headline'])
        clean_head = get_tokenized_lemmas(clean_head)
        docs.append(clean_head)
    tagged_docs = [TaggedDocument(d, [i]) for i, d in enumerate(docs)]
    #da cambiare i vari parametri
    model = Doc2Vec(tagged_docs, vector_size=vec_size, window=win_size, min_count=1, workers=-1, epochs = 50)
    return model


def doc_2_vec_extract(df,model):
    doc2vect_features = []
    for i, j in df.iterrows():
        clean_body = clean(j['articleBody'])
        clean_body = get_tokenized_lemmas(clean_body)
        vec_body = model.infer_vector(doc_words=clean_body)
        clean_head = clean(j['Headline'])
        clean_head = get_tokenized_lemmas(clean_head)
        vec_head = model.infer_vector(doc_words=clean_head)
        cos = 1 - spatial.distance.cosine(vec_body, vec_head)
        ls =np.concatenate((vec_body,vec_head,[cos]))
        doc2vect_features.append(ls)
    return doc2vect_features

