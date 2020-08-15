import os
from utils.freeze import *
from sklearn.externals import joblib
from textcnn_model.text_cnn import *


category_id = "651"
time_stamp = "1592366759"
model_prefix = "model-500"
label2id = joblib.load(f'model/runs_{category_id}/{time_stamp}/label2id.m')
with open(f'model_pb/id2label_{category_id}.txt', 'w') as f:
    for label in label2id:
        f.write(str(label2id[label]) + "\u0001" + label + '\n')
print(f"{category_id}类目数: {len(label2id)}")

model = TextCNN(sequence_length=10,
              num_classes=len(label2id),
              vocab_size=83144,
              embedding_size=64,
              filter_sizes=list(map(int, "3,4,5".split(","))),
              num_filters=128,
              l2_reg_lambda=0.1)

base_dir = f"model/runs_{category_id}/{time_stamp}/checkpoints"

ckpt_file = os.path.join(base_dir, model_prefix)
optimize_graph(model,
               ckpt_file=ckpt_file,
               input_tensors=[model.input_x, model.dropout_keep_prob],
               output_tensors=[model.scores, model.predictions],
               output_model_file=os.path.join("model_pb", f"model_{category_id}.pb"))

