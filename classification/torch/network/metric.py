

import os
import pandas
import numpy
from tensorflow.keras.callbacks import Callback
from sklearn import metrics


class metric(Callback):

    def __init__(self, generator=None, name=None, verbose=0, detail=[], report=[], folder=None):
        
        self.generator = generator
        self.name = name
        self.verbose = verbose
        self.detail = detail
        self.report = report
        self.folder = folder
        pass

    def on_epoch_end(self, epoch=None, logs=None):
        
        data = self.generator.flow()
        score = self.model.predict(data['feature'])
        prediction = score.argmax(axis=1)
        target = data['target'].argmax(axis=1)
        self.detail += [pandas.DataFrame({"score":score[:,1], "prediction":prediction, "target":target})]
        self.report += [summary(score[:,1], prediction, target, label=[1,0])]
        pass

        loss = metrics.log_loss(y_true=target, y_pred=score).round(3)
        accuracy = metrics.accuracy_score(y_true=target, y_pred=prediction).round(3)
        auc = metrics.roc_auc_score(y_score=score[:,1], y_true=target).round(3)
        pass

        logs['iteration'] = epoch#str(epoch+1).zfill(2)        
        logs["{} loss".format(self.name)] = loss
        logs["{} accuracy".format(self.name)] = accuracy
        logs["{} auc".format(self.name)] = auc
        if(self.verbose==1): print(logs)
        pass

    def save(self, what='detail', iteration=0):

        location = os.path.join(self.folder, '{}-detail.csv'.format(self.name))
        if(what=='detail'): self.detail[iteration].to_csv(location, index=False)
        if(what=='report'): write(self.report[iteration], to=os.path.join(self.folder, '{}-report.txt'.format(self.name)))
        return

    pass

def summary(score, prediction, target, label=None):

    # score = table['score']
    # prediction = table['prediction']# .replace({'lower':0, "higher":1})
    # target = table['target']#.replace({'lower':0, "higher":1})
    matrix = metrics.confusion_matrix(y_true=target, y_pred=prediction, labels=label)
    auc = metrics.roc_auc_score(y_score=score, y_true=target).round(2)
    tpr = (matrix[0,0] / sum(matrix[0,:])).round(2)
    tnr = (matrix[1,1] / sum(matrix[1,:])).round(2)
    ppv = (matrix[0,0] / sum(matrix[:,0])).round(2)
    npv = (matrix[1,1] / sum(matrix[:,1])).round(2)
    acc = (numpy.diag(matrix).sum() / matrix.sum()).round(2)
    output = {
        "matrix":matrix,
        "acc":acc,
        "auc":auc,
        "tpr":tpr,
        "tnr":tnr,
        "ppv":ppv,
        "npv":npv
    }
    return(output)

def write(text, to):

    folder = os.path.dirname(to)
    os.makedirs(folder, exist_ok=True)
    with open(to, 'wt') as paper: _ = paper.write(str(text))
    return