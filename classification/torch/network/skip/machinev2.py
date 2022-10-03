
# import os
# import numpy
# import pandas
# import tensorflow
# from tensorflow.keras.models import load_model
# from tensorflow.keras import metrics, optimizers, losses
# from tensorflow.keras.callbacks import ModelCheckpoint
# from sklearn.metrics import roc_auc_score
# from tensorflow.keras.callbacks import Callback
# from sklearn import metrics
# import shutil

# class machine:

#     def __init__(self, folder=None):
        
#         self.folder = folder
#         os.makedirs(self.folder, exist_ok=True) if(self.folder) else None
#         return

#     def load(self, what='model', path=None):

#         if(what=="model"):

#             # physical_devices = tensorflow.config.list_physical_devices('GPU') 
#             # tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)
#             self.model = load_model(path)
#             pass

#         return

#     def give(self, what='optimizer'):

#         if(what=='optimizer'):

#             self.optimizer = optimizers.Adam(learning_rate=1e-5)            
#             pass
        
#         if(what=='loss'):

#             self.loss = losses.BinaryCrossentropy()
#             pass
        
#         if(what=='checkpoint'):

#             # path = os.path.join(self.folder, '{val_loss:.2f}/')
#             # path = self.folder + "/{val_loss:.2f}/"
#             path = self.folder + "/{epoch:02d}/"
#             self.checkpoint = ModelCheckpoint(
#                 filepath=path,
#                 monitor="val_loss",
#                 verbose=0,
#                 save_best_only=False,
#                 save_weights_only=False,
#                 mode="auto",
#                 save_freq="epoch",
#                 options=None
#             )
#             pass

#         return
    
#     def save(self, what='history'):

#         if(what=='history'):

#             history = pandas.DataFrame(self.model.history.history)
#             history.to_csv(os.path.join(self.folder, "history.csv"), index=False)
#             pass

#         if(what=='better'):

#             iteration = pandas.DataFrame(self.model.history.history).sort_values(by=['validation auc'], ascending=False).reset_index(drop=True)['iteration'][0]
#             print("The {} iteration is better.".format(iteration))
#             selection = os.path.join(self.folder, iteration)
#             shutil.copytree(selection, os.path.join(self.folder, 'better'))
#             pass

#         return
    
#     def evaluate(self, generator=None, label=[1,0]):

#         data = generator.flow()
#         score = self.model.predict(data['feature'])
#         prediction = score.argmax(axis=1)
#         target = data['target'].argmax(axis=1)
#         track = {
#             "score":score[:,1],
#             "prediction":prediction,
#             "target":target
#         }
#         track = pandas.DataFrame(track)
#         pass
          
#         mat = metrics.confusion_matrix(y_true=target, y_pred=prediction, labels=label)
#         auc = metrics.roc_auc_score(y_score=score[:,1], y_true=target)
#         tpr = mat[0,0] / sum(mat[0,:])
#         tnr = mat[1,1] / sum(mat[1,:])
#         ppv = mat[0,0] / sum(mat[:,0])
#         npv = mat[1,1] / sum(mat[:,1])
#         acc = numpy.diag(mat).sum() / mat.sum()
#         output = {
#             "matrix":numpy.round(mat,3),
#             "acc":numpy.round(acc,3),
#             "auc":numpy.round(auc,3),
#             "tpr":numpy.round(tpr,3),
#             "tnr":numpy.round(tnr,3),
#             "ppv":numpy.round(ppv,3),
#             "npv":numpy.round(npv,3)
#         }
#         return(output, track)

#     pass


# class metric(Callback):

#     def __init__(self, generator):
        
#         self.generator = generator
#         pass

#     def on_epoch_end(self, epoch=None, logs=None):
        
#         data = self.generator.flow()
#         score = self.model.predict(data['feature'])
#         target = data['target'].argmax(axis=1)
#         auc = metrics.roc_auc_score(y_score=score[:,1], y_true=target)
#         logs["validation auc"] = auc
#         logs['iteration'] = str(epoch+1).zfill(2)
#         print('validation auc: {}'.format(auc))
#         pass

#     pass



# str(12).zfill(2)
# ##
# def perform(score, target, label):

#     prediction = score.argmax(axis=1)
#     mat = metrics.confusion_matrix(y_true=target, y_pred=prediction, labels=label)
#     acc = numpy.diag(mat).sum() / mat.sum()
#     auc = metrics.roc_auc_score(y_score=score[:,1], y_true=prediction)
#     tpr = mat[0,0] / sum(mat[0,:])
#     tnr = mat[1,1] / sum(mat[1,:])
#     ppv = mat[0,0] / sum(mat[:,0])
#     npv = mat[1,1] / sum(mat[:,1])
#     print("confuse matrix")
#     print(mat)
#     print('acc {}'.format(acc))
#     print('auc {}'.format(auc))
#     print('tpr {}'.format(tpr))
#     print('tnr {}'.format(tnr))
#     print('ppv {}'.format(ppv))
#     print('npv {}'.format(npv))
#     return(mat, acc, auc, tpr, tnr, ppv, npv)

# class auc(Callback):

#     def __init__(self, train, validation):

#         self.train = train
#         self.validation = validation
#         # self.train_x1 = train[0][0]
#         # self.train_x2 = train[0][1]
#         # self.train_y = train[1]
#         # self.validation_x1 = validation[0][0]
#         # self.validation_x2 = validation[0][1]
#         # self.validation_y = validation[1]
#         pass

#     def on_train_begin(self, logs={}):
#         return

#     def on_train_end(self, logs={}):
#         return

#     def on_epoch_begin(self, epoch, logs={}):
#         return

#     # def on_epoch_end(self, epoch, logs={}):
#     def on_epoch_end(self):
#         print("Go")
#         score = {
#             "train" : self.model.predict_proba(self.train[0]),
#             "validation" : self.model.predict_proba(self.validation[0])
#         }
#         target = {
#             "train" : numpy.argmax(self.train[1], axis=1),
#             "validation" : numpy.argmax(self.validation[1], axis=1)
#         }
#         result = {
#             'train' : round(roc_auc_score(y_score=score['train'][:,1], y_true=target['train']), 4),
#             "validation" : round(roc_auc_score(y_score=score['validation'][:,1], y_true=target['validation']), 4) 
#         }
#         # roc_train = roc_auc_score(y_score=score['train'][:,1], y_true=target['train'])
        
#         # roc_val = roc_auc_score(self.validation_y, y_pred_val)
#         message = 'train auc score: {}, validation auc score: {}'.format(result['train'], result['validation'])
#         print(message)
#         return

    # def on_batch_begin(self, batch, logs={}):
    #     return

    # def on_batch_end(self, batch, logs={}):
    #     return

# roc = RocCallback(training_data=(X_train, y_train),
#                   validation_data=(X_test, y_test))




# ##  載入當前模型版本
# model_path = "../resource/h5/v1/RiskClassifierModel.h5"
# model = load_model(model_path)
# # model.summary()

# ##  Optimizer 
# opt = optimizers.Adam(learning_rate=1e-5)

# ##  Complie model
# model.compile(
#     optimizer=opt,
#     loss="categorical_crossentropy"
#     # metrics=[metrics.BinaryAccuracy()]
# )

# ##  Set up checkpoint
# os.makedirs("../LOG/", exist_ok=True)
# filepath = "../LOG/{val_loss:.2f}/"
# # filepath = "../LOG/{val_loss:.2f}/"
# callback_checkpoint = ModelCheckpoint(
#     filepath,
#     monitor="val_loss",
#     verbose=0,
#     save_best_only=False,
#     save_weights_only=False,
#     mode="auto",
#     save_freq="epoch",
#     options=None
# )

# ##  Start training model
# epoch = 2
# batch = 8
# model.fit(
#     x=[train_img_array_bal, train_var_array_bal], y=train_target_array_bal, 
#     batch_size=batch, epochs=epoch, verbose=1,
#     callbacks=[callback_checkpoint], 
#     validation_split=0.0, validation_data=([val_img_array, val_var_array], val_target_array), shuffle=True,
#     class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None,
#     validation_steps=None, validation_batch_size=4, validation_freq=1,
#     max_queue_size=10, workers=1, use_multiprocessing=False
# )







# # model.fit(X_train, y_train, 
# #           validation_data=(X_test, y_test),
# #           callbacks=[roc])


