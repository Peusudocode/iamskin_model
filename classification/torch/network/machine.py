
import os
import numpy
import pandas
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics, optimizers, losses, backend
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from sklearn import metrics
import shutil

class machine:

    def __init__(self, folder=None):
        
        self.folder = folder
        os.makedirs(self.folder, exist_ok=True) if(self.folder) else None
        pass

        self.optimizer = optimizers.Adam(learning_rate=1e-5)
        self.loss = losses.BinaryCrossentropy()
        return

    def load(self, what='model', path=None):

        if(what=="model"): self.model = load_model(path)
        return

    def save(self, what='history'):

        if(what=='history'):

            history = pandas.DataFrame(self.model.history.history)
            history.to_csv(os.path.join(self.folder, "history.csv"), index=False)
            pass

        return
    
    pass
    
    # def evaluate(self, generator=None, label=None):

    #     data = generator.flow()
    #     score = self.model.predict(data['feature'])
    #     prediction = score.argmax(axis=1)
    #     target = data['target'].argmax(axis=1)
    #     track = {
    #         "score":score[:,1],
    #         "prediction":prediction,
    #         "target":target
    #     }
    #     track = pandas.DataFrame(track)
    #     pass
          
    #     mat = metrics.confusion_matrix(y_true=target, y_pred=prediction, labels=label)
    #     auc = metrics.roc_auc_score(y_score=score[:,1], y_true=target)
    #     tpr = mat[0,0] / sum(mat[0,:])
    #     tnr = mat[1,1] / sum(mat[1,:])
    #     ppv = mat[0,0] / sum(mat[:,0])
    #     npv = mat[1,1] / sum(mat[:,1])
    #     acc = numpy.diag(mat).sum() / mat.sum()
    #     output = {
    #         "matrix":numpy.round(mat,3),
    #         "acc":numpy.round(acc,3),
    #         "auc":numpy.round(auc,3),
    #         "tpr":numpy.round(tpr,3),
    #         "tnr":numpy.round(tnr,3),
    #         "ppv":numpy.round(ppv,3),
    #         "npv":numpy.round(npv,3)
    #     }
    #     return(output, track)

    # pass


    # def give(self, what='optimizer'):

    #     if(what=='optimizer'):

    #         self.optimizer = optimizers.Adam(learning_rate=5e-5)            
    #         pass
        
    #     if(what=='loss'):

    #         self.loss = losses.BinaryCrossentropy()
    #         pass
        
    #     if(what=='checkpoint'):

    #         # path = os.path.join(self.folder, '{val_loss:.2f}/')
    #         # path = self.folder + "/{val_loss:.2f}/"
    #         path = self.folder + "/{epoch:02d}/"
    #         self.checkpoint = ModelCheckpoint(
    #             filepath=path,
    #             monitor="val_loss",
    #             verbose=0,
    #             save_best_only=False,
    #             save_weights_only=False,
    #             mode="auto",
    #             save_freq="epoch",
    #             # save_freq=1500,
    #             options=None
    #         )
    #         pass

    #     return



        # if(what=='better'):

        #     iteration = pandas.DataFrame(self.model.history.history).sort_values(by=['{} auc'.format(name)], ascending=False).reset_index(drop=True)['iteration'][0]
        #     print("the {} iteration is better".format(iteration))
        #     selection = os.path.join(self.folder, iteration)
        #     shutil.copytree(selection, os.path.join(self.folder, 'better'))
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

