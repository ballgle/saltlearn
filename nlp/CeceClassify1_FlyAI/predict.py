# -*- coding: utf-8 -*
'''
实现模型的调用
'''
from flyai.dataset import Dataset
from keras.engine.saving import load_model
from model import Model
from path import MODEL_PATH
import os
from attention import AttLayer


class Predictor(object):
    def __init__(self, path=MODEL_PATH, name='final.h5'):
        self.data = Dataset()
        from keras.utils import CustomObjectScope
        with CustomObjectScope({'AttLayer': AttLayer}):
            self.model = load_model(os.path.join(path, name))

    def predict(self, **data):
        p = self.model.predict(self.data.predict_data(**data))
        return p

    def to_category(self, p):
        y = self.data.to_categorys(p)
        return y

    def predict_to_category(self, **data):
        p = self.predict(**data)
        y = self.to_category(p)
        return y
#1
# p = model.predict(MODEL_PATH, name='final.h5', text='我今晚上能见到他吗？')
if __name__ == '__main__':
    predictor = Predictor()
    #1
    y = predictor.predict_to_category(text='我今晚上能见到他吗？')
    print(y)

