

import os
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model, load_model
from tensorflow import random
from efficientnet import tfkeras
from tensorflow.python.keras.backend import var


##  Image and variable network model.
backbone = tfkeras.EfficientNetB7(weights='imagenet')
# backbone.summary()
image = {}
image['input'] = backbone.layers[0].input
image['output'] = backbone.layers[-2].output
image['model'] = Model(inputs=image['input'], outputs=image['output'])
variable = {}
variable['input'] = Input(shape=(16,), name='variable_input_default')
variable['output'] = variable['input']
variable['output'] = Dense( 64 , activation ='relu')(variable['output'])
variable['output'] = Dense(128 , activation ='relu')(variable['output'])
# variable['output'] = Dense(1024, activation ='relu')(variable['output'])
combination = {}
combination['input'] = concatenate([image['output'], variable['output']])
combination['output'] = combination['input']
# combination['output'] = Dense(512, activation ='relu')(combination['output'])
combination['output'] = Dense(128, activation ='relu')(combination['output'])
combination['output'] = Dense(  2, activation ='softmax')(combination['output'])
combination['model'] = Model([image['input'], variable['input']], combination['output'])
# combination['model'].summary()
model = combination['model']
folder = 'efficient/b7/'
os.makedirs(folder, exist_ok=True)
model.save(os.path.join(folder, '01.h5'), save_format='h5')
check = True
if(check):

    x = [random.uniform(shape=[4,600,600,3]), random.uniform(shape=[4,16])]
    model(x)
    pass

model.summary()









##  Image network model.
backbone = tfkeras.EfficientNetB7(weights='imagenet')
layer = backbone.layers
component = {
    "input":layer[0].input,
    "extraction":layer[-2].output
}
component['backbone'] = Model(inputs=component['input'], outputs=component['extraction'])
component['output'] = Dense(2, activation ='softmax')(component['backbone'].output)
model = Model(inputs=component['backbone'].input, outputs=component['output'])
# model.summary()
folder = 'efficient/b7/'
os.makedirs(folder, exist_ok=True)
model.save(os.path.join(folder, '02.h5'), save_format='h5')
# check = False
# if(check):

#     shape = [16]
#     shape += list(model.input.shape)[1:]
#     x = random.uniform(shape=shape)
#     model(x)
#     pass







# combination['layer'] = Sequential([
#     Dense(512, activation ='relu'),
#     Dense(256, activation ='relu'),
#     Dense(  2, activation ='softmax')
# ])
# combination['output'] = combination['layer'](combination['input'])



# image = {
#     "input":backbone.layers[0].input,
#     'model':Model(inputs=backbone.layers[0].input, outputs=backbone.layers[-2].output)
# }

# layer = {
#     "image term" : 
# }
# layer = backbone.layers
# component = {
#     "input":layer[0].input,
#     "extraction":layer[-2].output
# }
component['image term'] = Model(inputs=backbone.layers[0].input, outputs=backbone.layers[-2].output)
variable = {
    'layer':[
        Dense(256 , activation ='relu'),
        Dense(512 , activation ='relu'),
        Dense(1024, activation ='relu')
    ],
    'input': Input(shape=(16,), name='variable_input_default')
}
cache = variable['layer'][0](variable['input'])
cache = variable['layer'][1](cache)
output = variable['layer'][2](cache)
component['variable term'] = Model(inputs=variable['input'], outputs=output)
combination = {}
combination['size'] = list(component['variable term'].output.shape)[-1] + list(component['image term'].output.shape)[-1],
combination['input'] = Input(shape=combination['size'], name='variable_image_combination')
combination['layer'] = [
    Dense(512, activation='relu'),
    Dense(256, activation ='relu'),
    Dense(128, activation ='relu'),
    Dense(  2, activation ='softmax'),
]
cache = combination['layer'][0](combination['input'])
cache = combination['layer'][1](cache)
cache = combination['layer'][2](cache)
output = combination['layer'][3](cache)



component['image and variable term'] = Model(combination['input'], output)
Model([component['image term'].input, component['variable term'].input], component['image and variable term'].output)

component['variable term'].summary()

component['image and variable term']
# combination = list(component['variable term'].output.shape)[-1] + list(component['image term'].output.shape)[-1]




# variable = {}
# variable['input'] = Input(shape=(10,), name='variable_input_binary')
# variable['hidden 01'] = Dense(256, activation ='relu')
# variable['hidden 02'] = variable['hidden 01'](variable['input'])



component['output'] = Dense(2, activation ='softmax')(component['backbone'].output)
model = Model(inputs=component['backbone'].input, outputs=component['output'])
# model.summary()
folder = 'efficient/b7/'
os.makedirs(folder, exist_ok=True)
model.save(os.path.join(folder, '02.h5'), save_format='h5')

check = False
if(check):

    shape = [16]
    shape += list(model.input.shape)[1:]
    x = random.uniform(shape=shape)
    model(x)
    pass
# # path = 'resource/h5/v1/product/01.h5'
# # path = 'resource/h5/v1/densenet/01.h5'
# path = 'resource/h5/v1/resnet/01.h5'
# # path = 'resource/h5/v1/vgg/01.h5'
# core =  efn.EfficientNetB1(weights='imagenet')
# # core = load_model(path)
# core.summary()
# ex = -2
# core.layers[0].input
# core.layers[ex].output
# image_part = Model(inputs=core.layers[0].input, outputs=core.layers[ex].output)
# image_part.summary()


# variable = {}
# variable['input'] = Input(shape=(10,), name='variable_input_binary')
# variable['hidden 01'] = Dense(256, activation ='relu')
# variable['hidden 02'] = variable['hidden 01'](variable['input'])
# ##
# ##  Merge and output layer
# ptrBothHidden           = concatenate([core.layers[ex].output, variable['hidden 02']])
# ptrBothHiddenLayer_1 = Dense(256, activation ='relu')
# ptrBothHiddenLayer_2 = Dense(128, activation ='relu')
# ptrBothOutputLayer = Dense(2, activation ='softmax',
#                              kernel_regularizer = regularizers.l2(0.2),
#                              bias_regularizer = regularizers.l2(0.2),
#                              activity_regularizer = regularizers.l2(0.2))
# ptrBothHidden = ptrBothHiddenLayer_1(ptrBothHidden)
# ptrBothHidden = ptrBothHiddenLayer_2(ptrBothHidden)
# ptrBothOutput = ptrBothOutputLayer(ptrBothHidden)
# ptrModel      = Model(inputs=[core.layers[0].input, variable['input']], outputs=ptrBothOutput)
# ptrModel.summary()
# ptrModel.save('01.h5', save_format='h5')




# import tensorflow
# from tensorflow.keras import layers

# model = tensorflow.keras.Sequential()
# model.add(tensorflow.keras.Input(shape=(10,)))
# model.add(layers.Dense(256, activation="relu"))
# model.add(layers.Dense(512, activation="relu"))
# model.add(layers.Dense(256, activation="relu"))
# model.add(layers.Dense(128, activation="relu"))
# model.add(layers.Dense(64, activation="relu"))
# model.add(layers.Dense(2, activation="softmax"))
# model.save('01.h5', save_format='h5')



# import efficientnet as efn



# import efficientnet.tfkeras as efn
# model = efn.EfficientNetB0(weights='imagenet')
# model.summary()

# model.layers[-1].output

# import tensorflow
# x_1 = tensorflow.random.normal(shape=(12,224,224,3))
# x_2 = tensorflow.random.normal(shape=(12,10))
# ptrModel([x_1, x_2])


# m = load_model('cache.h5')
# m.summary()

# import numpy, datetime, os, pandas, random, pdb, sys, re, datetime, gc, shutil, timeit, tqdm, pickle, time
# import PIL.Image as pil
# import matplotlib.pyplot as plot
# from PIL import ImageFilter
# from tqdm import tqdm
# from sklearn import metrics
# from sklearn.model_selection import ParameterGrid
# from sklearn.model_selection import train_test_split as holdout
# from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, confusion_matrix,f1_score, average_precision_score
# import tensorflow
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.preprocessing.image import ImageDataGenerator as idg
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras import backend
# from tensorflow.keras.callbacks import Callback
# from tensorflow.keras.callbacks import ReduceLROnPlateau
# from tensorflow.keras.models import load_model
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.models import Model
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.layers import Input, BatchNormalization
# from tensorflow.keras import regularizers
# from tensorflow.keras import backend
# from tensorflow.keras.losses import categorical_crossentropy as logloss
# from tensorflow.keras.optimizers import Adadelta, Adam, SGD, RMSprop
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.applications.inception_v3 import InceptionV3
# from tensorflow.keras.applications.densenet import DenseNet121 as Dense121
# from tensorflow.keras.applications import ResNet50 as Res
# from tensorflow.keras.layers import GlobalAveragePooling2D
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, concatenate
# from tensorflow.keras.applications.vgg16 import VGG16


# class model:

#     def __init__(self, name, folder):
        
#         self.name = name
#         self.folder = folder
#         return

    

# def ResBothModel():
#     ##
#     ##  Image part
#     ptrPreTrainModel       = Res(include_top=True, weights="imagenet")
#     ptrImageInput  = ptrPreTrainModel.input
#     ptrImageHidden = ptrPreTrainModel.layers[-2].output
#     ##
#     ##  Variable part
#     ptrVariableInput = Input(shape=(16,))
#     ptrVariableHiddenLayer = Dense(256, activation ='relu')
#     ptrVariableHidden = ptrVariableHiddenLayer(ptrVariableInput)
#     ##
#     ##  Merge and output layer
#     ptrBothHidden           = concatenate([ptrImageHidden, ptrVariableHidden])
#     ptrBothHiddenLayer_1 = Dense(256, activation ='relu')
#     ptrBothHiddenLayer_2 = Dense(128, activation ='relu')
#     ptrBothOutputLayer = Dense(2, activation ='softmax',
#                                  kernel_regularizer = regularizers.l2(0.2),
#                                  bias_regularizer = regularizers.l2(0.2),
#                                  activity_regularizer = regularizers.l2(0.2))
#     ptrBothHidden = ptrBothHiddenLayer_1(ptrBothHidden)
#     ptrBothHidden = ptrBothHiddenLayer_2(ptrBothHidden)
#     ptrBothOutput      = ptrBothOutputLayer(ptrBothHidden)
#     ptrModel          = Model(inputs=[ptrImageInput, ptrVariableInput], outputs=ptrBothOutput)
#     return(ptrModel)
