# -*- coding: utf-8 -*
import sys

import argparse
import codecs
import json
import keras
import numpy as np
import os
from flyai.dataset import Dataset
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, LSTM, Dense, Embedding, Conv1D, Concatenate, BatchNormalization

import create_dict
import processor
from attention import AttLayer
from model import Model
from path import DATA_PATH
from path import MODEL_PATH

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

rnn_unit_1 = 100  # 第一层lstm包含cell个数
rnn_unit_2 = 100  # 第二层lstm包含cell个数
conv_dim = 128
embed_dim = 200
class_num = 2


def load_embed(char_dict):
    embed_path = os.path.join(DATA_PATH, 'embedding.json')
    with open(embed_path, encoding='utf-8') as jsonin:
        embed = json.load(jsonin)
    MAX_CHAR = max(char_dict.values())
    embed_mat = np.zeros((MAX_CHAR + 1, embed_dim))
    for c, v in embed.items():
        if c in char_dict and v != 'nan':
            embed_mat[char_dict[c], :] = v
    return embed_mat


# 数据获取辅助类
dataset = Dataset()
# 模型操作辅助类
model = Model(dataset)
# 超参
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")  # 默认为10
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")  # 默认为32
args = parser.parse_args()
MAX_LEN = processor.MAX_LEN

word_dict, word_dict_res = create_dict.load_dict()
num_word = max(word_dict.values()) + 1
# ——————————————————导入数据——————————————————————
embed_mat = load_embed(word_dict)
input_x = Input(shape=(MAX_LEN,), dtype='int32')
input_xc = Input(shape=(MAX_LEN,), dtype='int32')


def model_400M():
    def feed_input(input_x, sub_name):
        x1 = Embedding(input_dim=num_word, output_dim=embed_dim, name=sub_name + 'embed_s',
                       weights=[embed_mat], trainable=False)(input_x)
        x2 = Embedding(input_dim=num_word, output_dim=embed_dim, name=sub_name + 'embed_d',
                       weights=[embed_mat], trainable=True)(input_x)
        x = Concatenate()([x1, x2])
        # CNN model
        kls = [2, 3, 4, 5]
        hs = []
        for kl in kls:
            h = Conv1D(conv_dim, kl, activation='relu')(x)
            # h = GlobalMaxPool1D()(h)
            h = AttLayer()(h)
            hs.append(h)
        h2 = Concatenate()(hs)
        h2 = BatchNormalization()(h2)
        return h2

    h2 = feed_input(input_x, 'term')
    h2c = feed_input(input_xc, 'char')
    h2 = Concatenate()([h2, h2c])


# 训练集合大小为315364，大约1000*256
embed = Embedding(input_dim=num_word, output_dim=embed_dim, name='embed_s',
                  weights=[embed_mat], trainable=True)
x = embed(input_x)
h2_term = LSTM(rnn_unit_1, return_sequences=True)(x)
h2_term = AttLayer()(h2_term)
x_c = embed(input_xc)
h2_c = LSTM(rnn_unit_1, return_sequences=True)(x_c)
h2_c = AttLayer()(h2_c)
h2 = Concatenate()([h2_term, h2_c])

pred = Dense(class_num, activation='softmax')(h2)
k_model = keras.Model([input_x, input_xc], pred)
opt = keras.optimizers.Adam(0.001)
k_model.compile(opt, 'categorical_crossentropy', ['acc', ])

earlystop = EarlyStopping(min_delta=0.01, patience=1)
save_best = ModelCheckpoint(os.path.join(MODEL_PATH, "model.h5"), save_best_only=True)

save_epochs = 100
best_loss = 1e6
patient = 2
patient_count = patient
for epochs in range(args.EPOCHS):
    if epochs == args.EPOCHS - 1 or (epochs != 0 and epochs % save_epochs == 0):
        x_train, y_train, x_test, y_test = dataset.next_batch(args.BATCH, test_data=True)
        k_model.fit(x_train, y_train, batch_size=args.BATCH, verbose=1)
        score = k_model.evaluate(x_test, y_test, verbose=1, batch_size=args.BATCH)
        print('val score', score)
        score = score[0]  # val loss
        if score <= best_loss:
            model.save_model(k_model, MODEL_PATH, name='model.h5', overwrite=True)
            patient_count = patient
        else:
            patient_count -= 1
        if patient_count < 0:
            break
    else:
        x_train, y_train, x_test, y_test = dataset.next_batch(args.BATCH, test_data=False)
        verbose = 1 if epochs % 10 == 0 else 0
        k_model.fit(x_train, y_train, batch_size=args.BATCH, verbose=verbose)
    if epochs % 10 == 0:
        print(str(epochs) + "/" + str(args.EPOCHS))
