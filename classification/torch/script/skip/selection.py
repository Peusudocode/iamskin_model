

##
import data, network, extension
import shutil, os


##
tabulation = data.tabulation()
tabulation.read(path="resource/csv/0922/standard information.csv")
tabulation.split(what='train', size=0.8, target='vote')
tabulation.split(what='validation', size=1, target='vote')
# tabulation.load(
#     train='resource/csv/1101/train_new.csv', 
#     validation="resource/csv/1101/validation.csv"
# )
print(tabulation.validation.shape)
print(tabulation.validation.head())


##
batch = 32
output = 'variable'
engine = 'default'
generator = {
    # "train" : data.generator(table=tabulation.train, batch=batch, mode='train', output=output),
    "validation" : data.generator(table=tabulation.validation, batch=batch, mode='validation', output=output, engine=engine)
    # "test" : data.generator(table=tabulation.test, batch=batch, mode='test'),
}


##
record = ''
folder = "LOG/VARIABLE"
best   = None
iteration = extension.drive.parse(folder)
for path, do in iteration:
    
    if(do):

        ##  載入指定模型的 checkpoint ，針對驗證集進行預測，驗證結果存在模型 checkpoint 資料夾。 
        machine = network.machine()
        machine.load(what='model', path=path)
        evaluation, validation = machine.evaluate(generator=generator['validation'], label=[1,0])
        validation.to_csv(path + '/validation.csv', index=False)
        pass

        ##  將評估結果存在模型 checkpoint 資料夾。
        evaluation['path'] = path
        extension.drive.write(text=str(evaluation), to=evaluation['path'] + '/evaluation.txt')
        pass

        ##  保留整體上述的紀錄。
        message = "path {} | auc {}".format(evaluation['path'], evaluation['auc'])
        record += message + '\n'
        pass
        
        ##  留下在驗證集表現較好的 checkpoint 模型。
        if(best==None):

            best = {"path":evaluation['path'], "auc":evaluation['auc']}
            print("best in {} | auc {}".format(best['path'], best['auc']))
            pass

        else:
            
            ##  下一個 checkpoint 沒有當前的好，就將其資料夾刪除。
            if(best['auc'] >= evaluation['auc']):

                extension.drive.remove(evaluation['path'])
                print("best in {} | auc {}".format(best['path'], best['auc']))
                pass
            
            ##  下一個 checkpoint 比當前的好，將當前的資料夾刪除，由後者取代。
            else:

                extension.drive.remove(best['path'])
                best = {"path":evaluation['path'], "auc":evaluation['auc']}
                print("best in {} | auc {}".format(best['path'], best['auc']))
                pass

            pass
        
        pass

    pass


##  將整體結果記錄以及最好的模型資訊記錄下來。
extension.drive.write(text=str(record), to=folder+'/record.txt')
extension.drive.write(text=str(best), to=folder+'/best.txt')
#best['save path'] = os.path.join(os.path.dirname(best['path']), "better")
# best['save path'] = folder + '/' + 'better'
extension.drive.copy(best['path'], folder + '/' + 'better')
# shutil.copytree(best['path'], best['save path'])