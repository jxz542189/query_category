import os
import collections
import pandas as pd
import tensorflow as tf
from termcolor import colored
from sklearn.utils import shuffle
from sklearn.externals import joblib
from utils.data_helpers import *
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.model_fn import EstimatorSpec
from tensorflow.contrib import learn


basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
logger = set_logger(colored('train', 'cyan'), False)

label2id = joblib.load(os.path.join(basedir, 'transformer_model', 'model', 'label2id.m'))
print(label2id)
id2label = {}
for label, id in label2id.items():
    id2label[id] = label
with open(os.path.join(basedir, 'data', 'root_id2label.txt'), 'w') as f:
    for label, id in label2id.items():
        f.write(str(id) + "\t" + label + "\n")
print(id2label)