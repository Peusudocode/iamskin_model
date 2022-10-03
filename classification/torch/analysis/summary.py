

from sklearn import metrics
import numpy
import json

def summary(target=None, score=None, prediction=None, label=[0,1], save=False):

    matrix = metrics.confusion_matrix(y_true=target, y_pred=prediction, labels=label)
    tpr = (matrix[0,0] / sum(matrix[0,:])).round(2)
    tnr = (matrix[1,1] / sum(matrix[1,:])).round(2)
    ppv = (matrix[0,0] / sum(matrix[:,0])).round(2)
    npv = (matrix[1,1] / sum(matrix[:,1])).round(2)
    acc = (numpy.diag(matrix).sum() / matrix.sum()).round(2)
    auc = metrics.roc_auc_score(y_score=score, y_true=target).round(2)
    output = {
        "matrix":matrix,
        "acc":acc,
        "auc":auc,
        "tpr":tpr,
        "tnr":tnr,
        "ppv":ppv,
        "npv":npv
    }
    if(save):

        with open('review output.txt', 'w') as paper:

            print(output, file=paper)
            pass

        pass

    return(output)



