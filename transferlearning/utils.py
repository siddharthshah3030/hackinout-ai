import os
import tensorflow as tf
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense


class TransferLearning():
    def __init__(self):
        self.X = []
        self.train_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            fill_mode='nearest',
            vertical_flip=True,
            horizontal_flip=True
        )
        self.batch_size = 32
        self.steps_per_epoch = 32
        self.epochs = 20

    
    def get_json_data_from_server(self, project_id):
        api_base = "http://34.67.52.131:1337"
        r = requests.get(api_base+"/projects/"+project_id)
        
        return r.json()


    def get_data(self, json):
        data = json['classes']['classes']

        for d in data:
            for item in d['data']:
                self.X.append({'image': item['url'], 'label': d['name']})

        return self.X


    def get_data_frame(self, X):
        df = pd.DataFrame(data = X)
        df['path'] = df['image'].apply(lambda x: x.replace('https://storage.googleapis.com/deployml/', ''))
        return df

    def load_image(self, imagepath, shape=(224,224)):
        print(imagepath, '------------')
        response = requests.get(imagepath)
        img = Image.open(BytesIO(response.content))
        img = img.resize(shape)
        return img


    def get_resnet_model(self, num_classes=2):
        self.model = Sequential()
        self.model.add(ResNet50(include_top = False, pooling = 'avg', weights = 'imagenet'))
        self.model.add(Dense(num_classes, activation = 'softmax'))
        self.model.layers[0].trainable = False

        return self.model

    def save_images(self, df, shape=(224, 224)):
        self.X = []
        self.y = []
        for index, row in df.iterrows():
            # if not os.path.exists("../tmp/"+row['label']+"/" + row['path'].split('/')[0]):
            #     os.makedirs("../tmp/"+row['label']+"/" + row['path'].split('/')[0])
            
            print(index, row)
            image = self.load_image(row['image'], shape=shape)
            self.X.append(np.array(image))
            self.y.append(row['label'])
            # image.save("../tmp/"+row['label']+"/"+row['path'])

        return self.X, self.y

    def split_data_set(self, X, y, test_size=0.20, shuffle=True):
        X = np.array(X)
        y = np.asarray(y)

        encoder = LabelBinarizer()
        y_cat = encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.20, shuffle=True)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        return X_train, X_test, y_train, y_test

    def compile_model(self, loss='binary_crossentropy'):
        sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
        self.model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])

    def train(self):
        self.history = self.model.fit_generator(self.train_datagen.flow(self.X_train, self.y_train, batch_size=self.batch_size),
                              steps_per_epoch=self.steps_per_epoch, epochs=self.epochs,
                              validation_data=(self.X_test, self.y_test))
    
    def do_transfer_learning(self, project_id):
        # TODO: project wise customisations
        print("Getting data from server -------------\n")
        json = self.get_json_data_from_server(project_id)
        print("Parse json -------------\n")
        data = self.get_data(json)
        print("Dataframe -------------\n")
        df = self.get_data_frame(data)

        print("Preprocessing and Saving Images -------------\n")
        # TODO: project wise shape
        X, y = self.save_images(df)


        print("Split Data -------------\n")
        X_train, X_test, y_train, y_test = self.split_data_set(X, y)

        self.model = self.get_resnet_model(1)
        self.compile_model()
        print("Training the model -------------\n")
        self.train()

        print("Saving the model -------------\n")

        self.model.save()