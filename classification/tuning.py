
##  The packages.
import os
import pandas
import numpy 
# import tensorflow
from sklearn import metrics
from tensorflow import keras

##  Constant of configuration.
dataset = 'acne04-single'
batch_size   = 32
train_dir  = './resource/{}/train'.format(dataset)
val_dir    = './resource/{}/validation'.format(dataset)
test_dir   = './resource/{}/test'.format(dataset)
image_shape      = (224, 224)
epochs         = 10
seen_class_num = 4
class_name     = ['levle0', 'levle1', 'levle2', 'levle3']

##  Create data generator function.
def create_ImageDataGenerator(data_dir, batch_size, shuffle=True):

    image_gen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=keras.applications.resnet.preprocess_input
    )
    data_gen = image_gen.flow_from_directory(
            batch_size=batch_size,
            directory=data_dir,
            shuffle=shuffle,
            color_mode="rgb",
            target_size=image_shape,
            class_mode='categorical',
            seed = 42
        )

    return(data_gen)

##  Create data generator respectively.
image_gen_train = create_ImageDataGenerator(data_dir = train_dir, batch_size=batch_size,shuffle=True)
image_gen_val   = create_ImageDataGenerator(data_dir = val_dir  , batch_size=batch_size,shuffle=False)
image_gen_test  = create_ImageDataGenerator(data_dir = test_dir , batch_size=batch_size,shuffle=False)

##
def create_model(pretrained_model='resnet'):

    if(pretrained_model=='resnet'):
        
        base_model = keras.applications.resnet.ResNet101(weights = 'imagenet', include_top = False)
        # hidden_size = base_model.output.shape[-1]
        hidden_out = base_model.output 
        hidden_out = keras.layers.GlobalAveragePooling2D()(hidden_out) # 2048 # -3
        hidden_out = keras.layers.Dense(1024, activation='relu')(hidden_out) # -2 
        predictions = keras.layers.Dense(seen_class_num, activation='softmax')(hidden_out) # -1
        model = keras.models.Model(inputs=base_model.input, outputs=predictions)
        pass

    ##  Todo
    ##  使用 Densenet or InceptionV3, etc.
    return(model)

model = create_model(pretrained_model='resnet')

optimizer = keras.optimizers.Adam(
    learning_rate=0.0001, 
    beta_1=0.9, beta_2=0.999, 
    epsilon=None, decay=0.0, amsgrad=False
)
model.compile(
    optimizer=optimizer, 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

early_stopping  = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
step_size_train = image_gen_train.n // image_gen_train.batch_size
step_size_val  = image_gen_val.n // image_gen_val.batch_size

model.fit(
    image_gen_train,
    epochs = epochs,
    validation_data=image_gen_val,
    callbacks = [early_stopping]    
)

def get_summary(image_gen):

    n_batches = len(image_gen)
    y_true = numpy.concatenate([numpy.argmax(image_gen[i][1], axis=1) for i in range(n_batches)])
    y_score = numpy.argmax(model.predict(image_gen, steps=n_batches), axis=1) 
    con_matrix = metrics.confusion_matrix(y_true, y_score)
    con_matrix = pandas.DataFrame(con_matrix)
    con_report = metrics.classification_report(y_true, y_score)
    return(con_matrix, con_report)

val_con_matrix, val_con_report = get_summary(image_gen_val)
print('-'*10, 'validation', '-'*10)
print(val_con_matrix)
print(val_con_report)

test_con_matrix, test_con_report = get_summary(image_gen_test)
print('-'*10, 'test', '-'*10)
print(test_con_matrix)
print(test_con_report)

##
base_model_name = "resnet"
os.makedirs("./model/{}/".format(dataset), exist_ok=True)
model.save('./model/{}/FineTune[{}]Classifier.h5'.format(dataset, base_model_name))
backbone_layer_model = keras.Model(model.inputs, model.layers[-3].output)
backbone_layer_model.save('./model/{}/FineTune[{}]Backbone.h5'.format(dataset, base_model_name))


