
##  The packages.
import data
import network
import analysis

'''
導入訓練好的模型，在整個資料集上做測試，這部份不區分訓練集、測試集。
目前在於讓客戶理解哪些 case 會確實分對，哪些 case 會分錯，
給用戶一個預期心裡。
'''

##  Load data, then split to train and test.
tabulation = data.tabulation()
tabulation.read(path='resource/20211231v2/csv/information.csv')

##  Data loader.
batch = 36
output = 'image and variable'
engine = 'default'
generator = {
    "data" : data.generator(table=tabulation.data, batch=batch, mode='data', output=output, engine=engine)
}
machine = network.machine(folder=None)
path = [
    './resource/20211231v2/h5/mole-release-20211231v2-official/2/DENSENET IMAGE/better',
    './resource/20211231v2/h5/mole-release-20211231v2-official/2/RESNET IMAGE/better',
    './resource/20211231v2/h5/mole-release-20211231v2-official/3/RESNET IMAGE/better',
    './resource/20211231v2/h5/mole-release-20211231v2-official/4/RESNET IMAGE/better',
    './resource/20211231v2/h5/mole-release-20211231v2-official/4/RESNET DEFAULT IMAGE AND VARIABLE/better'
]

##  Preidction.
import tqdm
import pandas
score = dict()
for number, p in enumerate(path, 1):

    print("="*5, "load [{}] model".format(p), "="*5)
    machine.load(what='model', path=p)
    score[str(number)] = []
    for iteration, batch in tqdm.tqdm(enumerate(generator['data']), total=len(generator['data'])):
        
        (image, variable), target = batch
        target = target[:,1].tolist()
        if('VARIABLE' in p): score[str(number)] += machine.model.predict([image, variable])[:,1].tolist()
        if('VARIABLE' not in p): score[str(number)] += machine.model.predict([image])[:,1].tolist()
        pass
    pass

summary = pandas.DataFrame(score)
summary['mean score'] = summary.mean(1)
summary['update prediction'] = 1*(summary['mean']>0.5)
summary['target'] = tabulation.data['target']
summary['image_crop'] = tabulation.data['image_crop']
analysis.summary(
    target=summary['target'],
    score=summary['mean'],
    prediction=summary['prediction'],
    label=[1,0], 
    save=False
)
summary['previous prediction'] = tabulation.data['prediction'].replace({"lower":0, "higher":1})
analysis.summary(
    target=summary['previous prediction'],
    score=summary['mean score'],
    prediction=summary['update prediction'],
    label=[1,0], 
    save=False
)
