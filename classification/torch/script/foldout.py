
##  The packages.
import data
import network

'''
將資料切分成訓練集、測試集，根據訓練集執行 K-fold 交叉驗證。
'''

##  Load data, then split to train and test.
tabulation = data.tabulation()
tabulation.read(path='resource/20220119v2/csv/information.csv')
tabulation.split(test=0.2, target='target')
tabulation.test.to_csv('resource/20220119v2/csv/test.csv', index=False)
tabulation.train.to_csv('resource/20220119v2/csv/train.csv', index=False)

##  K-fold block.
fold = data.fold(data=tabulation.train, k=4, target='target')

##  Each fold block.
for b in [1,2,3,4]:
    
    train, validation = fold.select(block=b)

    ##  Batch generator function.
    batch = 8
    output = 'image and variable'
    engine = 'default'
    generator = {
        "train" : data.generator(table=train, batch=batch, mode='train', output=output, engine=engine),
        # "validation" : data.generator(table=validation, batch=batch, mode='validation', output=output, engine=engine),
        "test" : data.generator(table=tabulation.test, batch=batch, mode='test', output=output, engine=engine),
    }

    ##  Model setting.
    network.memory()
    folder = "LOG/{}/MOBILENET IMAGE AND VARIABLE/".format(str(b))
    machine = network.machine(folder=folder)
    machine.load(what='model', path='./resource/20220119v2/h5/acne-release-prototype/mobilenet-default-image-and-variable-model')
    machine.model.compile(optimizer=machine.optimizer, loss=machine.loss)

    ##  Learning process.
    epoch  = 20
    weight = None
    metric = {
        # 'validation':network.metric(generator=generator['validation'], name='validation', verbose=0, detail=[], report=[], folder=folder),
        'test':network.metric(generator=generator['test'], name='test', verbose=1, detail=[], report=[], folder=folder)
    }
    # checkpoint = network.checkpoint(folder=folder, track='validation auc')
    checkpoint = network.checkpoint(folder=folder, track='test auc')
    runtime = network.runtime()
    machine.model.fit(
        generator['train'], epochs=epoch, verbose=1, 
        # callbacks=[metric['validation'], metric['test'], checkpoint], 
        callbacks=[metric['test'], checkpoint], 
        validation_steps=None, class_weight=weight,
        workers=1, shuffle=True, initial_epoch=0
    )
    runtime.touch()

    ##  Save the history and the better model.
    machine.save(what='history')
    # metric['validation'].save(what='detail', iteration=checkpoint.iteration)
    # metric['validation'].save(what='report', iteration=checkpoint.iteration)
    metric['test'].save(what='detail', iteration=checkpoint.iteration)
    metric['test'].save(what='report', iteration=checkpoint.iteration)
    pass

