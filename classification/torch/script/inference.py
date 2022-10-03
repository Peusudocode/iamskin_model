
import pandas
import os
from sklearn import metrics
import plotnine

'''
任意 Fold 的模型在測試集上做推論的結果整理。
'''

##  Ensemble on the test.
# folder = 'resource/20220119v2/h5/acne-release-20220216-official/'
folder = './LOG/'
name = 'test-detail.csv'
path = [
    os.path.join(folder, '1', "DENSENET IMAGE AND VARIABLE", name),
    os.path.join(folder, '1', 'RESNET IMAGE AND VARIABLE', name),
    os.path.join(folder, '1', 'MOBILENET IMAGE AND VARIABLE', name),
    os.path.join(folder, '1', 'VGGNET IMAGE AND VARIABLE', name),
    # os.path.join(folder, '1', 'DEFAULT VARIABLE', name),
    # os.path.join(folder, '2', 'DEFAULT VARIABLE', name),
    # os.path.join(folder, '3', 'DEFAULT VARIABLE', name),
    # os.path.join(folder, '4', 'DEFAULT VARIABLE', name),
    # os.path.join(folder, '1', "DENSENET DEFAULT IMAGE AND VARIABLE", name),
    # os.path.join(folder, '2', "DENSENET DEFAULT IMAGE AND VARIABLE", name),
    # os.path.join(folder, '3', "DENSENET DEFAULT IMAGE AND VARIABLE", name),
    # os.path.join(folder, '4', "DENSENET DEFAULT IMAGE AND VARIABLE", name),
    # os.path.join(folder, '1', 'DENSENET IMAGE', name),
    # os.path.join(folder, '2', 'DENSENET IMAGE', name),
    # os.path.join(folder, '3', 'DENSENET IMAGE', name),
    # os.path.join(folder, '4', 'DENSENET IMAGE', name),
    # os.path.join(folder, '1', 'RESNET DEFAULT IMAGE AND VARIABLE', name),
    # os.path.join(folder, '2', 'RESNET DEFAULT IMAGE AND VARIABLE', name),
    # os.path.join(folder, '3', 'RESNET DEFAULT IMAGE AND VARIABLE', name),
    # os.path.join(folder, '4', 'RESNET DEFAULT IMAGE AND VARIABLE', name),
    # os.path.join(folder, '1', 'RESNET IMAGE', name),
    # os.path.join(folder, '2', 'RESNET IMAGE', name),
    # os.path.join(folder, '3', 'RESNET IMAGE', name),
    # os.path.join(folder, '4', 'RESNET IMAGE', name),
    # os.path.join(folder, '1', 'MOBILENET DEFAULT IMAGE AND VARIABLE', name),
    # os.path.join(folder, '2', 'MOBILENET DEFAULT IMAGE AND VARIABLE', name),
    # os.path.join(folder, '3', 'MOBILENET DEFAULT IMAGE AND VARIABLE', name),
    # os.path.join(folder, '4', 'MOBILENET DEFAULT IMAGE AND VARIABLE', name),
    # os.path.join(folder, '1', 'MOBILENET IMAGE', name),
    # os.path.join(folder, '2', 'MOBILENET IMAGE', name),
    # os.path.join(folder, '3', 'MOBILENET IMAGE', name),
    # os.path.join(folder, '4', 'MOBILENET IMAGE', name)
    # os.path.join(folder, '1', 'VGGNET DEFAULT IMAGE AND VARIABLE', name),
    # os.path.join(folder, '2', 'VGGNET DEFAULT IMAGE AND VARIABLE', name),
    # os.path.join(folder, '3', 'VGGNET DEFAULT IMAGE AND VARIABLE', name),
    # os.path.join(folder, '4', 'VGGNET DEFAULT IMAGE AND VARIABLE', name),
    # os.path.join(folder, '1', 'VGGNET IMAGE', name),
    # os.path.join(folder, '2', 'VGGNET IMAGE', name),
    # os.path.join(folder, '3', 'VGGNET IMAGE', name),
    # os.path.join(folder, '4', 'VGGNET IMAGE', name)
    # os.path.join(folder, '1', 'PRODUCT IMAGE AND VARIABLE', name),
    # os.path.join(folder, '2', 'PRODUCT IMAGE AND VARIABLE', name),
    # os.path.join(folder, '3', 'PRODUCT IMAGE AND VARIABLE', name),
    # os.path.join(folder, '4', 'PRODUCT IMAGE AND VARIABLE', name)    
]
for i, p in enumerate(path, 1):

    d = pandas.read_csv(p)
    print(p)
    print(metrics.classification_report(y_pred=d['prediction'], y_true=d['target']))
    if(i==1):

        score = d['score']
        target = d['target']
        pass

    else:

        score = score + d['score']
        target = target + d['target']
        pass

    pass

threshold = 0.5
score = score / i
target = target / i
prediction = 1 * (score > threshold)
detail = pandas.DataFrame({'score':score, 'target':target, 'prediction':prediction})
pass

##  Select models should change the location folder name.
# location = os.path.join(folder, 'ensemble-top-5')
# os.makedirs(location, exist_ok=True)
# detail.to_csv(os.path.join(location, name), index=False)
# pass

##  Plot the AUC graph on the test.
# path += [os.path.join(location, name)]
# group = []
# for p in path:

#     item = pandas.read_csv(p)
#     fpr, tpr, threshold = metrics.roc_curve(y_true=item['target'], y_score=item['score'])
#     auc = metrics.auc(fpr, tpr)
#     title = "auc: {}".format(round(auc, 3)) + " | " + str.replace(p, folder, "")
#     group += [pandas.DataFrame({"fpr":fpr, "tpr":tpr, "threshold":threshold, "auc":auc, "title":title})]
#     pass
fpr, tpr, threshold = metrics.roc_curve(y_true=detail['target'], y_score=detail['score'])
auc = metrics.auc(fpr, tpr)
print(auc)

# data = pandas.concat(group, axis=0)
# paper = plotnine.ggplot()
# paper = paper + plotnine.geom_line(plotnine.aes(x=data['fpr'], y=data['tpr'], colour=data['title']), linetype='solid') 
# plotnine.ggsave(plot = paper, filename = name + " auc plot.png", path = location)