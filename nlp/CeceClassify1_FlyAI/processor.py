# -*- coding: utf-8 -*
import jieba
import os
from flyai.processor.base import Base

from path import DATA_PATH  # 导入输入数据的地址

jieba.load_userdict(os.path.join(DATA_PATH, 'keywords'))
import numpy as np
import create_dict

MAX_LEN = 100


class Processor(Base):
    # 该参数需要与app.yaml的Model的input-->columns->name 一一对应
    def __init__(self):
        super(Processor, self).__init__()
        self.word_dict, self.word_dict_res = create_dict.load_dict()

    def input_x(self, text):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        if type(text) is not str:
            with open('data/err_text.txt', 'a', encoding='utf-8') as fout:
                fout.write('{}\n'.format(text))
            text = str(text)
        terms = jieba.cut(text, cut_all=False)
        truncate_terms = []
        for term in terms:
            truncate_terms.append(term)
            if len(truncate_terms) >= MAX_LEN:
                break
        index_list = [self.word_dict[term] if term in self.word_dict
                      else create_dict._UNK_ for term in truncate_terms]
        if len(index_list) < MAX_LEN:
            index_list = index_list + [create_dict._PAD_] * (MAX_LEN - len(index_list))

        char_index_list = [self.word_dict[c] if c in self.word_dict
                           else create_dict._UNK_ for c in text]
        char_index_list = char_index_list[:MAX_LEN]
        if len(char_index_list) < MAX_LEN:
            char_index_list = char_index_list + [create_dict._PAD_] * (MAX_LEN - len(char_index_list))
        return index_list, char_index_list

    # 该参数需要与app.yaml的Model的output-->columns->name 一一对应
    def input_y(self, gid):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''
        y = np.zeros(2)
        label_index = int(gid)
        y[label_index] = 1
        return y

    def output_y(self, data):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        out_y = np.argmax(data)
        return out_y
