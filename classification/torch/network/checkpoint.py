

from tensorflow.keras.callbacks import Callback
from sklearn import metrics
import os


class checkpoint(Callback):

    def __init__(self, folder=None, track=None):
        
        self.folder = folder
        self.track = track
        pass

    def on_epoch_end(self, epoch=None, logs=None):

        if(epoch==0): 

            '''第一個 epoch 會需要初始化一個 history 用於後續存放其他 epoch 的結果。'''
            self.history = {}
            for k in logs: self.history.update({k:[logs[k]]})
            pass

            '''第一個 epoch 結束後儲存模型，並建立一個 better 用於後續 epoch 比較。'''
            location = os.path.join(self.folder, 'better')
            self.model.save(location)
            self.better = logs[self.track]
            self.iteration = logs['iteration']
            pass

        else:

            '''將 epoch 結果累積放入 history 。'''
            for k in logs: self.history[k] += [logs[k]]
            pass

            '''檢查是否有變好，如果有進入更新流程。'''
            if(self.better < logs[self.track]):

                '''儲存較好的模型，取代先前的檔案，並更新 better 用於後續 epoch 比較。'''
                print('save the better since {} > {}'.format(logs[self.track], self.better))
                location = os.path.join(self.folder, 'better')
                self.model.save(location)
                self.better = logs[self.track]
                self.iteration = logs['iteration']
                pass

            pass

        pass

    pass