
import pandas, numpy
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

class tabulation:

    def __init__(self):
        
        return

    def read(self, path):

        self.data = pandas.read_csv(path)
        return

    def split(self, test=0.2, target=None):

        train, test = train_test_split(self.data, test_size=test, random_state=0, stratify=self.data[target])
        self.train = train.reset_index(drop=True).copy()
        self.test = test.reset_index(drop=True).copy()
        return

    pass

        # if(name=='train'): self.train = pandas.read_csv(path)#.sample(200).reset_index(drop=True)
        # if(name=='validation'): self.validation = pandas.read_csv(path)
        # if(name=='test'): self.test = pandas.read_csv(path)

    # def fold(self, k, target=None):

    #     self.k = k
    #     self.data['block'] = None
    #     knife = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    #     for number, (_, index) in enumerate(knife.split(self.data, self.data[target]), 1):
            
    #         self.data.at[index, 'block'] = number
    #         pass
        
    #     return
    
    # def select(self, block=1):

    #     train = self.data.loc[self.data['block']!=block].copy()
    #     validation = self.data.loc[self.data['block']==block].copy()
    #     return(train, validation)

    # pass



# import pandas, numpy
# from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

# class tabulation:

#     def __init__(self):
        
#         return

#     def read(self, path):

#         self.data = pandas.read_csv(path)
#         return

#     def split(self, what='train', size=None, target=None):

#         if(what=='train'):

#             if(size<1):

#                 train, data = train_test_split(self.data, test_size=1-size, random_state=0, stratify=self.data[target])
#                 self.train = train
#                 self.data = data
#                 pass

#             else:

#                 self.train = self.data
#                 self.data = None
#                 pass
        
#         if(what=='test'):

#             if(size<1):

#                 test, data = train_test_split(self.data, test_size=1-size, random_state=0, stratify=self.data[target])
#                 self.test = test
#                 self.data = data
#                 pass

#             else:

#                 self.test = self.data
#                 self.data = None
#                 pass

#         if(what=='validation'):

#             if(size<1):

#                 validation, data = train_test_split(self.data, test_size=1-size, random_state=0, stratify=self.data[target])
#                 self.validation = validation
#                 self.data = data
#                 pass

#             else:

#                 self.validation = self.data
#                 self.data = None
#                 pass

#             pass

#     def load(self, train=None, validation=None):

#         self.train = pandas.read_csv(train) if(train) else None
#         self.validation = pandas.read_csv(validation) if(validation) else None
#         pass    

#     def fold(self, k, target=None):

#         self.k = k
#         self.data['block'] = None
#         knife = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
#         for number, (_, index) in enumerate(knife.split(self.data, self.data[target]), 1):
            
#             self.data.at[index, 'block'] = number
#             pass
        
#         return
    
#     def choose(self, block=1):

#         self.train = self.data.loc[self.data['block']!=block].copy()
#         self.validation = self.data.loc[self.data['block']==block].copy()
#         return

#     pass
