# -*- coding: utf-8 -*

import os
from flyai.model.base import Base
from keras.engine.saving import load_model

from path import MODEL_PATH

KERAS_MODEL_NAME = "model.h5"
from attention import AttLayer


class Model(Base):
    def __init__(self, data):
        self.data = data

    def predict(self, **data):
        from keras.utils import CustomObjectScope
        with CustomObjectScope({'AttLayer': AttLayer}):
            path = MODEL_PATH
            name = KERAS_MODEL_NAME
            model = load_model(os.path.join(path, name))
        data = model.predict(self.data.predict_data(**data))

        return self.data.to_categorys(data)

    def predict_all(self, datas):
        from keras.utils import CustomObjectScope
        with CustomObjectScope({'AttLayer': AttLayer}):
            path = MODEL_PATH
            name = KERAS_MODEL_NAME
            model = load_model(os.path.join(path, name))
        outputs = []
        for data in datas:
            output = model.predict(self.data.predict_data(**data))
            output = self.data.to_categorys(output)
            outputs.append(output)
        return outputs

    def save_model(self, model, path, name=KERAS_MODEL_NAME, overwrite=False):
        super().save_model(model, path, name, overwrite)
        model.save(os.path.join(path, name))
