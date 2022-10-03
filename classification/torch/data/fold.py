

import pandas, numpy
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

class fold:

    def __init__(self, data=None, k=None, target=None):
        
        self.data = data
        self.k = k
        self.data['block'] = None
        knife = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
        loop  = enumerate(knife.split(self.data, self.data[target]), 1)
        for number, (_, index) in loop: self.data.loc[index, 'block'] = number
        return
    
    def select(self, block=1):

        train = self.data.loc[self.data['block']!=block].copy()
        validation = self.data.loc[self.data['block']==block].copy()
        return(train, validation)
    
    pass
