
# ##  Val confusion matrix.
# n_batches = len(image_gen_val)
# val_con_matrix = metrics.confusion_matrix(
#     numpy.concatenate([numpy.argmax(image_gen_val[i][1], axis=1) for i in range(n_batches)]),
#     numpy.argmax(model.predict(image_gen_val, steps=n_batches), axis=1) 
# )
# val_con_matrix = pandas.DataFrame(val_con_matrix)

# ##  View the val result.
# val_report = metrics.classification_report(
#     numpy.concatenate([numpy.argmax(image_gen_val[i][1], axis=1) for i in range(n_batches)]), 
#     numpy.argmax(model.predict(image_gen_val, steps=n_batches), axis=1), 
#     target_names=class_name
# )
# print(val_report)

# ##  Test confusion matrix.
# n_batches = len(image_gen_test)
# test_con_matrix = metrics.confusion_matrix(
#     numpy.concatenate([numpy.argmax(image_gen_test[i][1], axis=1) for i in range(n_batches)]),
#     numpy.argmax(model.predict(image_gen_test, steps=n_batches), axis=1) 
# )
# test_con_matrix = pandas.DataFrame(test_con_matrix)

# ##  The test report.
# test_report = metrics.classification_report(
#     numpy.concatenate([numpy.argmax(image_gen_test[i][1], axis=1) for i in range(n_batches)]), 
#     numpy.argmax(model.predict(image_gen_test, steps=n_batches), axis=1), 
#     target_names=class_name
# )
# print(test_report)

##  Save the original find-tuned model.
# model.save('./model/{}/FineTuneResNet101_original.h5'.format(dataset))



##  2022/08/01
##  Checkpoint

# image_gen_val = ImageDataGenerator(preprocessing_function=preprocess_input)
# val_data_gen = image_gen_val.flow_from_directory(
#     batch_size=batch_size,
#     directory=val_dir,
#     target_size=(IMG_SHAPE, IMG_SHAPE),
#     class_mode='categorical',
#     color_mode="rgb",
#     seed = 42
# )

# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report

# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.applications.resnet import ResNet101, preprocess_input
# from tensorflow.keras.applications import DenseNet121, MobileNetV2
# # from tensorflow.keras.applications.efficientnet import EfficientNetB0
# # from efficientnet.tfkeras import EfficientNetB0
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
 

# '''
# 給定 pre-trained 模型, 針對特定資料集執行 fine-tuning 訓練. 
# '''





##  Train generator.

# train_data_gen.class_indices
# batch = next(iter(train_data_gen))
# x, y = batch
# x.shape
# y.shape

##  Val generator.
# image_gen_val = ImageDataGenerator(preprocessing_function=preprocess_input)
# val_data_gen = image_gen_val.flow_from_directory(
#     batch_size=batch_size,
#     directory=val_dir,
#     target_size=(IMG_SHAPE, IMG_SHAPE),
#     class_mode='categorical',
#     color_mode="rgb",
#     seed = 42
# )

##  Test generator.
# image_gen_test = ImageDataGenerator(preprocessing_function=preprocess_input)
# test_data_gen = image_gen_test.flow_from_directory(
#     batch_size=batch_size,
#     directory=test_dir,
#     target_size=(IMG_SHAPE, IMG_SHAPE),
#     class_mode='categorical',
#     color_mode="rgb",
#     seed = 42
# )


##  Model without output layer.
# base_model_defualt = ResNet101(weights = 'imagenet')
# base_model_defualt
# base_model_defualt.summary()

# base_model = ResNet101(weights = 'imagenet', include_top = False)
# base_model.summary()
# base_model_name = "resnet"

# base_model = DenseNet121(weights = 'imagenet', include_top = False)
# base_model_name = 'DenseNet121'
# base_model = EfficientNetB0(weights = 'imagenet', include_top = False)
# base_model_name = 'EfficientNetB0'
# base_model = MobileNetV2(weights = 'imagenet', include_top = False)
# base_model_name = 'MobileNetV2'

##  Lock the layers of model.
# for layer in base_model.layers: layer.trainable = False
# x.shape
# base_model(x).shape
# x_demo = GlobalAveragePooling2D()(base_model(x))
# x_demo.shape

##  Add a global averge polling layer.
# x = base_model.output 
# x = GlobalAveragePooling2D()(x) # 2048 # -3

# ##  Add a dense layer.
# x = Dense(1024, activation='relu')(x) # -2 

# ##  Add a classifier.
# predictions = Dense(seen_class_num, activation='softmax')(x) # -1

# ##  Constructure
# model = Model(inputs=base_model.input, outputs=predictions)

##  Optimizer.
# optimizer = Adam(
#     learning_rate=0.0001, beta_1=0.9, beta_2=0.999, 
#     epsilon=None, decay=0.0, amsgrad=False
# )


# ##  Complie.
# model.compile(
#     optimizer=optimizer, 
#     loss='categorical_crossentropy', 
#     metrics=['accuracy']
# )

# ##  Early stop setting
# early_stopping  = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
# STEP_SIZE_TRAIN = train_data_gen.n // train_data_gen.batch_size
# STEP_SIZE_VALID = val_data_gen.n // val_data_gen.batch_size

##
# model.fit(
#     train_data_gen,
#     steps_per_epoch=STEP_SIZE_TRAIN,
#     epochs = epochs,
#     validation_data=val_data_gen,
#     validation_steps=STEP_SIZE_VALID,
#     # class_weight=class_weights,
#     callbacks = [early_stopping]    
# )

## --- 0729 checkpoint

# n_batches = len(image_gen_train)
# x1 = numpy.concatenate([numpy.argmax(image_gen_train[i][1], axis=1) for i in range(n_batches)])
# x2 = numpy.concatenate([numpy.argmax(image_gen_train[i][1], axis=1) for i in range(n_batches)])


# ##  Val confusion matrix.
# n_batches = len(image_gen_val)
# val_con_matrix = confusion_matrix(
#     numpy.concatenate([numpy.argmax(image_gen_val[i][1], axis=1) for i in range(n_batches)]),
#     numpy.argmax(model.predict(val_data_gen, steps=n_batches), axis=1) 
# )
# val_con_matrix = pandas.DataFrame(val_con_matrix)

# ##  View the val result.
# val_report = classification_report(
#     numpy.concatenate([numpy.argmax(val_data_gen[i][1], axis=1) for i in range(n_batches)]), 
#     numpy.argmax(model.predict(val_data_gen, steps=n_batches), axis=1), 
#     target_names=class_name
# )
# print(val_report)

# ##  Test confusion matrix.
# n_batches = len(test_data_gen)
# test_con_matrix = confusion_matrix(
#     numpy.concatenate([numpy.argmax(test_data_gen[i][1], axis=1) for i in range(n_batches)]),
#     numpy.argmax(model.predict(test_data_gen, steps=n_batches), axis=1) 
# )
# test_con_matrix = pandas.DataFrame(test_con_matrix)

# ##  The test report.
# test_report = classification_report(
#     numpy.concatenate([numpy.argmax(test_data_gen[i][1], axis=1) for i in range(n_batches)]), 
#     numpy.argmax(model.predict(test_data_gen, steps=n_batches), axis=1), 
#     target_names=class_name
# )
# print(test_report)

# ##  Save the original find-tuned model.
# # model.save('./model/{}/FineTuneResNet101_original.h5'.format(dataset))

# ##
# os.makedirs("./model/{}/".format(dataset), exist_ok=True)
# model.save('./model/{}/FineTune[{}]Classifier.h5'.format(dataset, base_model_name))

# ##
# new_model = Model(model.inputs, model.layers[-3].output)
# new_model.save('./model/{}/FineTune[{}]Backbone.h5'.format(dataset, base_model_name))



# from tensorflow.keras import models
# test_load_model = models.load_model('./model/{}/FineTuneResNet101-feature.h5'.format(dataset))
# test_load_model.summary()

# model.layers.pop()
# model.summary()