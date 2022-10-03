
import pandas
import os
from sklearn import metrics
import plotnine

'''
指定特定的 Fold ，比較不同模型在 Validation 的表現，包含 ensemble 的結果。
'''

##  For each fold, ensemble on the test and validation.
# folder = 'resource/20220119v2/h5/acne-release-20220119v2-research/'
folder = './LOG/'
block = '1'
name = 'test-detail.csv'
path = [
    os.path.join(folder, block, "DENSENET IMAGE AND VARIABLE", name),
    os.path.join(folder, block, 'RESNET IMAGE AND VARIABLE', name),
    os.path.join(folder, block, 'MOBILENET IMAGE AND VARIABLE', name),
    os.path.join(folder, block, 'VGGNET IMAGE AND VARIABLE', name),
    # os.path.join(folder, block, 'DEFAULT VARIABLE', name),
    # os.path.join(folder, block, "DENSENET DEFAULT IMAGE AND VARIABLE", name),
    # os.path.join(folder, block, 'DENSENET IMAGE', name),
    # os.path.join(folder, block, 'RESNET DEFAULT IMAGE AND VARIABLE', name),
    # os.path.join(folder, block, 'RESNET IMAGE', name),
    # os.path.join(folder, block, 'MOBILENET DEFAULT IMAGE AND VARIABLE', name),
    # os.path.join(folder, block, 'MOBILENET IMAGE', name)
]
for i, p in enumerate(path, 1):

    d = pandas.read_csv(p)
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

location = os.path.join(folder, block, 'ensemble')
os.makedirs(location, exist_ok=True)
detail.to_csv(os.path.join(location, name), index=False)
pass

##  For each fold, plot the AUC graph on the test and validation.
path += [os.path.join(folder, block, 'ensemble', name)]
group = []
for p in path:

    item = pandas.read_csv(p)
    fpr, tpr, threshold = metrics.roc_curve(y_true=item['target'], y_score=item['score'])
    auc = metrics.auc(fpr, tpr)
    title = "auc: {}".format(round(auc, 3)) + " | " + str.replace(p, folder, "")
    # title = str.replace(p, os.path.join(folder, block), "").split('/')[1]
    group += [pandas.DataFrame({"fpr":fpr, "tpr":tpr, "threshold":threshold, "auc":auc, "title":title})]
    pass

data = pandas.concat(group, axis=0)
paper = plotnine.ggplot()
paper = paper + plotnine.geom_line(plotnine.aes(x=data['fpr'], y=data['tpr'], colour=data['title']), linetype='solid') 
plotnine.ggsave(plot = paper, filename = name + " auc plot.png", path = os.path.join(folder, block))

