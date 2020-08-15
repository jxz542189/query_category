import re
import os
import logging
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.utils import shuffle


basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(data_file, is_train=True, level=3):
    df = pd.read_csv(data_file, header=None)
    df = shuffle(df)
    records = df.to_dict('records')
    x_text = []
    labels = []
    y = []
    filename = os.path.join(basedir, 'data/label2id.m')

    for record in records:
        if level == 3:
            if len(str(record[1])) >= 5:
                x_text.append(str(record[0]).lower())
                labels.append(str(record[1]))
        else:
            if len(str(record[1])) == 3:
                x_text.append(str(record[0]).lower())
                labels.append(str(record[1]))

    if is_train:
        label2id = {}
        for id, label in enumerate(set(labels)):
            label2id[label] = id
        num_classes = len(set(labels))
        joblib.dump(label2id, filename)
    else:
        label2id = joblib.load(filename)
        num_classes = len(label2id)

    for label in labels:
        id =label2id[label]
        y.append(id)

    return x_text, y, label2id, num_classes


def load_data_and_labels_v1(data_file, label2id, level=3):
    df = pd.read_csv(data_file, header=None)
    df = shuffle(df)
    records = df.to_dict('records')
    x_text = []
    labels = []
    y = []

    for record in records:
        if level == 3:
            if len(str(record[1])) >= 5:
                x_text.append(str(record[0]).lower())
                labels.append(str(record[1]))
        else:
            if len(str(record[1])) == 3:
                x_text.append(str(record[0]).lower())
                labels.append(str(record[1]))

    for label in labels:
        id = label2id[label]
        y.append(id)

    return x_text, y, len(label2id)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # if shuffle:
        #     shuffle_indices = np.random.permutation(np.arange(data_size))
        #     shuffled_data = data[shuffle_indices]
        # else:
        shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def set_logger(context, verbose=False):
    if os.name == 'nt':  # for Windows
        return NTLogger(context, verbose)

    logger = logging.getLogger(context)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        '%(levelname)-.5s:' + context + ':[%(filename).10s:%(funcName).20s:%(lineno)3d]:%(message)s', datefmt=
        '%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger


class NTLogger:
    def __init__(self, context, verbose):
        self.context = context
        self.verbose = verbose

    def info(self, msg, **kwargs):
        print('I:%s:%s' % (self.context, msg), flush=True)

    def debug(self, msg, **kwargs):
        if self.verbose:
            print('D:%s:%s' % (self.context, msg), flush=True)

    def error(self, msg, **kwargs):
        print('E:%s:%s' % (self.context, msg), flush=True)

    def warning(self, msg, **kwargs):
        print('W:%s:%s' % (self.context, msg), flush=True)

def id2label(file_name, sep='\t'):
    label2id = {}
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            line = re.sub('\n', '', line)
            if len(line.split(sep)) == 2:
                id, label = line.split(sep)
                label2id[label] = int(id)
    return label2id


def read_word2id(file_name):
    word2id = {}
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            line = re.sub('\n', '', line)
            if len(line.split(' ')) == 2:
                word, id = line.split(' ')
                word2id[word] = int(id)
    return word2id


def process(line, word2id, max_sequence_length=15):
    res = [0] * max_sequence_length

    words = line.split(' ')[:max_sequence_length]
    for i, word in enumerate(words):
        if word in word2id:
            res[i] = word2id[word]
        else:
            res[i] = 0

    return res


def char_process(line, char2id, max_sequence_length=10, max_chars=15):
    res = [[0] * max_chars] * max_sequence_length

    words = line.split(' ')[:max_sequence_length]
    for i, word in enumerate(words):
        for j, c in enumerate(word[:max_chars]):
            if c in char2id:
                res[i][j] = char2id[c]
            else:
                res[i][j] = 0

    return res

if __name__ == '__main__':
    x_text, y, label2id, num_classes = load_data_and_labels("/Users/jxz/PycharmProjects/query_classify/data/train_level_3_20200421.csv")
    print(x_text[:5])
    print(basedir)
