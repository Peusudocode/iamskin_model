

import numpy
import os
from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


constant = {}
constant['image size'] = 224
constant['image folder'] = 'resource/20220119v2/jpg/default/'


class generator(Sequence):
    
    def __init__(self, table, batch=32, mode='train', output='image and variable', engine='default'):
        
        self.table = table
        self.batch = batch
        self.mode = mode
        self.output = output
        self.engine = engine
        return
    
    ##  一個 epoch 有多少 batch 數量。
    def __len__(self):

        number = numpy.floor(len(self.table) / self.batch)
        number = int(number)
        return(number) 

    ##  組成一個 batch 資料。
    def __getitem__(self, index):

        start = index * self.batch
        end = (index+1) * self.batch
        batch = self.table.iloc[start:end,:]
        batch = self.generate(batch=batch)
        return(batch)

    ##  當 epoch 結束時，針對 train 洗牌。
    def on_epoch_end(self):

        self.table = shuffle(self.table) if(self.mode =='train') else self.table
        pass

    ##  處理一個 batch 資料。
    def generate(self, batch):

        if(self.engine=='default'):

            var_term = numpy.empty((self.batch, 27))
            img_term = numpy.empty((self.batch, constant['image size'], constant['image size'], 3))
            target_term = numpy.empty((self.batch, 2), dtype=int)
            pass
        
        # if(self.engine=='binary'):

        #     var_term = numpy.empty((self.batch, 10))
        #     img_term = numpy.empty((self.batch, constant['image size'], constant['image size'], 3))
        #     target_term = numpy.empty((self.batch, 2), dtype=int)
        #     pass

        ##  輸入 batch 是一個 pandas 結構
        for iteration, (_, item) in enumerate(batch.iterrows()):
            # print(item['image_crop'])
            ##  處理變數
            var_term[iteration,] = process.variable(item=item, engine=self.engine) if('variable' in self.output) else None

            ##  處理圖片
            img_term[iteration,:,:,:] = process.image(item, mode=self.mode) if('image' in self.output) else None

            ##  處理標記
            target_term[iteration,:] = process.target(item)
            pass
        
        output = None
        output = ((img_term, var_term), target_term) if(self.output=='image and variable') else output
        output = (img_term, target_term) if(self.output=='image') else output
        output = (var_term, target_term) if(self.output=='variable') else output
        return(output)

    ##  處理全部資料。
    def flow(self):

        if(self.engine=='default'):

            var_term = numpy.empty((len(self.table), 27))
            img_term = numpy.empty((len(self.table), constant['image size'], constant['image size'], 3))
            target_term = numpy.empty((len(self.table), 2), dtype=int)
            pass

        # if(self.engine=='binary'):
        
        #     var_term = numpy.empty((len(self.table), 10))
        #     img_term = numpy.empty((len(self.table), constant['image size'], constant['image size'], 3))
        #     target_term = numpy.empty((len(self.table), 2), dtype=int)
        #     pass

        ##  輸入 batch 是一個 pandas 結構
        for iteration, (_, item) in enumerate(self.table.iterrows()):

            ##  處理變數
            var_term[iteration,] = process.variable(item=item, engine=self.engine) if('variable' in self.output) else None

            ##  處理圖片
            img_term[iteration,:,:,:] = process.image(item, mode=self.mode) if('image' in self.output) else None

            ##  處理標記
            target_term[iteration,:] = process.target(item)
            pass
        
        output = None
        output = {'feature':(img_term, var_term), "target":target_term} if(self.output=='image and variable') else output
        output = {'feature':img_term, "target":target_term} if(self.output=='image') else output
        output = {'feature':var_term, "target":target_term} if(self.output=='variable') else output
        return(output)
    pass


##
##
class process:

    def variable(item, engine='default'):
        
        if(engine=='default'):

            partition = (0,0,0,0,0)
            partition = (1,0,0,0,0) if((item['partition']=='額頭區域')) else partition
            partition = (0,1,0,0,0) if((item['partition']=='鼻子區域')) else partition
            partition = (0,0,1,0,0) if((item['partition']=='臉頰區域')) else partition
            partition = (0,0,0,1,0) if((item['partition']=='嘴巴、下巴區域')) else partition
            partition = (0,0,0,0,1) if((item['partition']=='我不確定')) else partition
            None if(partition!=(0,0,0,0,0)) else print("partition get missing value")
            howlong = (0,0)
            howlong = (1,0) if((item['howlong']=='三個月以下')) else howlong
            howlong = (0,1) if((item['howlong']=='三個月以上')) else howlong
            None if(howlong!=(0,0)) else print("howlong get missing value")
            squeeze = (0,0)
            squeeze = (1,0) if((item['squeeze']=='有')) else squeeze
            squeeze = (0,1) if((item['squeeze']=='沒有')) else squeeze
            None if(squeeze!=(0,0)) else print("squeeze get missing value")
            cream = (0,0,0,0)
            cream = (1,0,0,0) if(item['cream']=='有，醫師開立的外用藥') else cream
            cream = (0,1,0,0) if(item['cream']=='有，藥局購買的外用藥') else cream
            cream = (0,0,1,0) if(item['cream']=='有，非藥用抗痘產品') else cream
            cream = (0,0,0,1) if(item['cream']=='沒有') else cream
            None if(cream!=(0,0,0,0)) else print("cream get missing value")
            medicine = (0,0)
            medicine = (1,0) if(item['medicine']=='是') else medicine
            medicine = (0,1) if(item['medicine']=='否') else medicine
            None if(medicine!=(0,0)) else print("medicine get missing value")
            age = (0,0,0,0,0)
            age = (1,0,0,0,0) if((item['age']=='18歲以下')) else age
            age = (0,1,0,0,0) if((item['age']=='19-25歲')) else age
            age = (0,0,1,0,0) if((item['age']=='26-40歲')) else age
            age = (0,0,0,1,0) if((item['age']=='41-50歲')) else age
            age = (0,0,0,0,1) if((item['age']=='51歲以上')) else age
            None if(age!=(0,0,0,0,0)) else print("age get missing value")
            sex = (0,0,0)
            sex = (1,0,0) if((item['sex']=='男性')) else sex
            sex = (0,1,0) if((item['sex']=='女性')) else sex
            sex = (0,0,1) if((item['sex']=='不想回答')) else sex
            None if(sex!=(0,0,0)) else print("sex get missing value")
            menstruation = (0,0,0,0)
            menstruation = (1,0,0,0) if((item['menstruation']=='是')) else menstruation
            menstruation = (0,1,0,0) if((item['menstruation']=='否')) else menstruation
            menstruation = (0,0,1,0) if((item['menstruation']=='我不確定')) else menstruation
            menstruation = (0,0,0,1) if((item['menstruation']=='性別男性跳過')) else menstruation
            None if(menstruation!=(0,0,0,0)) else print("menstruation get missing value")
            encode = numpy.array(partition + howlong + squeeze + cream + medicine + age + sex + menstruation)
            pass
        
        return(encode)

    def image(item, mode):
        
        folder = constant['image folder']
        size = (constant['image size'], constant['image size'])
        link = os.path.join(folder, 'risk', item['image_default'])
        # link = os.path.join(folder, item['image'])
        picture = load_img(link).resize(size)
        picture = img_to_array(picture)
        picture = transform(picture) / 255 if(mode=='train') else picture / 255
        return(picture)

    def target(item):

        target = None
        target = [1,0] if(item['target']==0) else target
        target = [0,1] if(item['target']==1) else target
        return(target)


##  輸入 image 是 array 型態。 
def transform(image):

    generator = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False, 
        featurewise_std_normalization=False, 
        samplewise_std_normalization=False, 
        zca_whitening=False, 
        zca_epsilon=1e-06, 
        rotation_range=360,  ## 
        width_shift_range=[-10, 10], 
        height_shift_range=[-10, 10], 
        brightness_range=[1-0.5, 1+0.5], ## (-0.5,0.5)
        shear_range=0.5, 
        zoom_range=[1, 1+0.2],  ## (-0.5,0.5)
        channel_shift_range=0.1, ##  0.5
        fill_mode='nearest', 
        cval=0.0,
        horizontal_flip=True, ## 
        vertical_flip=True, ## 
        rescale=None, 
        preprocessing_function=None, 
        data_format=None, 
        validation_split=0.0, 
        dtype=None     
    )
    output = next(generator.flow(numpy.expand_dims(image, 0))).squeeze()
    return(output)    