
import os
import pandas
import plotnine
from sklearn import metrics

##  For each fold, plot the AUC graph on the test and validation.
# path += [os.path.join(folder, block, 'ensemble', name)]
folder = '/LOG/1/VOTE/test-detail.csv'
group = []
for p in path:

    item = pandas.read_csv(p)
    fpr, tpr, threshold = metrics.roc_curve(y_true=item['target'], y_score=item['score'])
    auc = metrics.auc(fpr, tpr)
    title = "auc: {}".format(round(auc, 3)) + " | " + str.replace(p, folder, "")
    group += [pandas.DataFrame({"fpr":fpr, "tpr":tpr, "threshold":threshold, "auc":auc, "title":title})]
    pass

data = pandas.concat(group, axis=0)
paper = plotnine.ggplot()
paper = paper + plotnine.geom_line(plotnine.aes(x=data['fpr'], y=data['tpr'], colour=data['title']), linetype='solid') 
plotnine.ggsave(plot = paper, filename = name + " auc plot.png", path = os.path.join(folder, block))

