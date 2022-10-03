
from tabulate import tabulate
from sklearn import metrics
import numpy
import pandas

'''指定資料集去檢視當前模型的輸出結果與醫師的標記。'''

def summary(score, prediction, target, label=None):

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

##  Review overall data.
table = pandas.read_csv("resource/20220119v2/csv/information.csv")
score = table['score']
prediction = table['prediction']
target = table['target']
report = summary(score=score, prediction=prediction, target=target, label=[1,0])
print(report)
print(table['vote'].value_counts())
print(table['partition'].value_counts())
print(table['howlong'].value_counts())
print(table['squeeze'].value_counts())
print(table['cream'].value_counts())
print(table['medicine'].value_counts())
print(table['age'].value_counts())
print(table['sex'].value_counts())
print(table['menstruation'].value_counts())

print(tabulate(pandas.crosstab(table['vote'], table['partition']), headers='keys'))
print(tabulate(pandas.crosstab(table['vote'], table['partition'], normalize='index'), headers='keys'))
print(tabulate(pandas.crosstab(table['vote'], table['partition'], normalize='columns'), headers='keys'))
print(tabulate(pandas.crosstab(table['vote'], table['howlong']), headers="keys"))
print(tabulate(pandas.crosstab(table['vote'], table['howlong'], normalize='index'), headers="keys"))
print(tabulate(pandas.crosstab(table['vote'], table['howlong'], normalize='columns'), headers="keys"))
print(tabulate(pandas.crosstab(table['vote'], table['squeeze']), headers="keys"))
print(tabulate(pandas.crosstab(table['vote'], table['squeeze'], normalize='index'), headers="keys"))
print(tabulate(pandas.crosstab(table['vote'], table['squeeze'], normalize='columns'), headers="keys"))
print(tabulate(pandas.crosstab(table['vote'], table['cream']), headers="keys"))
print(tabulate(pandas.crosstab(table['vote'], table['cream'], normalize='index'), headers="keys"))
print(tabulate(pandas.crosstab(table['vote'], table['cream'], normalize='columns'), headers="keys"))
print(tabulate(pandas.crosstab(table['vote'], table['medicine']), headers="keys"))
print(tabulate(pandas.crosstab(table['vote'], table['medicine'], normalize='index'), headers="keys"))
print(tabulate(pandas.crosstab(table['vote'], table['medicine'], normalize='columns'), headers="keys"))
print(tabulate(pandas.crosstab(table['vote'], table['age']), headers="keys"))
print(tabulate(pandas.crosstab(table['vote'], table['age'], normalize='index'), headers="keys"))
print(tabulate(pandas.crosstab(table['vote'], table['age'], normalize='columns'), headers="keys"))
print(tabulate(pandas.crosstab(table['vote'], table['sex']), headers="keys"))
print(tabulate(pandas.crosstab(table['vote'], table['sex'], normalize='index'), headers="keys"))
print(tabulate(pandas.crosstab(table['vote'], table['sex'], normalize='columns'), headers="keys"))
print(tabulate(pandas.crosstab(table['vote'], table['menstruation']), headers="keys"))
print(tabulate(pandas.crosstab(table['vote'], table['menstruation'], normalize='index'), headers="keys"))
print(tabulate(pandas.crosstab(table['vote'], table['menstruation'], normalize='columns'), headers="keys"))
pass

##  Review train data.
table = pandas.read_csv("resource/20220119v2/csv/train.csv")
score = table['score']
prediction = table['prediction']
target = table['target']
report = summary(score=score, prediction=prediction, target=target, label=[1,0])
print(report)

##  Review test data.
table = pandas.read_csv("resource/20220119v2/csv/test.csv")
score = table['score']
prediction = table['prediction']
target = table['target']
report = summary(score=score, prediction=prediction, target=target, label=[1,0])
print(report)

