

'''沒在用'''

##  The packages.
import data, network, extension


##  Load data.
tabulation = data.tabulation()
tabulation.read(path='resource/20211117/beta/csv/information.csv')
tabulation.fold(k=5, target='target')
tabulation.choose(block=1)
tabulation.validation = tabulation.validation.sample(20)

##  Prepare the data for learning.
batch = 16
output = 'image'
engine = 'default'
generator = {
    "train" : data.generator(table=tabulation.train, batch=batch, mode='train', output=output, engine=engine),
    "validation" : data.generator(table=tabulation.validation, batch=batch, mode='validation', output=output, engine=engine)
    # "test" : data.generator(table=tabulation.test, batch=batch, mode='test'),
}
# machine.model.predict(generator['validation'].flow()['feature'])

##  Model setting.
folder = "LOG (BETA)/IMAGE (RESNET)/"
# network.session(rate=1)
machine = network.machine(folder=folder)
machine.load(what='model', path='./resource/20211117/beta/h5/version-1.0.0/resnet-image-model')
machine.give(what='optimizer')
machine.give(what='loss')
machine.give(what='checkpoint')
machine.model.compile(
    optimizer=machine.optimizer,
    loss=machine.loss
)


##  Learning process.
epoch  = 20
weight = None
metric = network.metric(generator['validation'])
runtime = extension.runtime()
machine.model.fit(
    generator['train'], epochs=epoch, verbose=1, 
    callbacks=[metric, machine.checkpoint], 
    # validation_data=generator['validation'], 
    validation_steps=None, 
    class_weight=weight, max_queue_size=10, 
    workers=1, use_multiprocessing=False, 
    shuffle=True, initial_epoch=0
)
runtime.touch()


##  Save the history and the better model.
machine.save(what='history')
machine.save(what='better')


##  Load the better model and evaluate the validation,
##  save the prediction and report of validation.
machine.load(what='model', path=folder + '/better')
report, score = machine.evaluate(generator=generator['validation'])
score.to_csv(folder + '/better/validation.csv', index=False)
extension.drive.write(report, folder+'/better report.txt')


##  Store the better model only.
for i in range(1, epoch+1):

    extension.drive.remove(folder + '/' + str(i).zfill(2))
    pass


