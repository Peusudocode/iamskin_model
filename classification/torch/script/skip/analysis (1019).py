

import data, network, extension


'''
紀錄時間：2021/10/15
變數 `change` 原先有三種選項，分別是「無變化」、「不記得」、「有變化」，將「不記得」與「無變化」進行合併，二值化。
變數 `mole_size` 原為二值化，不動。
變數 `period` 將「不記得」與「一年以上」合併」,「其他」與「一年以下」合併，二值化。
變數 `gender` 將「不想回答」與「女性」合併，二值化。
變數 `age` 將 66/65 歲以下的選項合併，二值化。
變數二值化轉換後，個別與標記結果進行比對。
'''


##
tabulation = data.tabulation()
tabulation.load(train='resource/csv/1101/train.csv', validation='resource/csv/1101/validation.csv')
# tabulation.read(path='resource/csv/information.csv')
# tabulation.data['mole_size'].value_counts()
# tabulation.split(what='train', size=0.8, target='vote')
# tabulation.split(what='validation', size=1, target='vote')
# print(tabulation.validation.head())
#         dataset  image                            user_id  ...                                      image_default                                         image_crop download
# 186   MoleMe_OA    247  Ub4a64e317ed3fb2d0802b35ad77900b4  ...  Ub4a64e317ed3fb2d0802b35ad77900b4-daiecb5h-201...  Ub4a64e317ed3fb2d0802b35ad77900b4-daiecb5h-201...     True
# 488   MoleMe_OA    774  Uf9eb586ae977e6bc3a2c59b6e38b4816  ...  Uf9eb586ae977e6bc3a2c59b6e38b4816-6rl8lxok-201...  Uf9eb586ae977e6bc3a2c59b6e38b4816-6rl8lxok-201...     True
# 2498  MoleMe_OA   6223  Uc6573af29fabbb341ebd03c7732cad0a  ...  Uc6573af29fabbb341ebd03c7732cad0a-ok2___ey-202...  Uc6573af29fabbb341ebd03c7732cad0a-ok2___ey-202...     True
# 1953     MoleMe    107  U56d301fdb98fc437c5bacf62693bd55b  ...  U56d301fdb98fc437c5bacf62693bd55b-202105201143...  U56d301fdb98fc437c5bacf62693bd55b-202105201143...     True
# 2265  MoleMe_OA   5894  U7ff08586a8c5baf8600a7ce0310a8c11  ...  U7ff08586a8c5baf8600a7ce0310a8c11-4sgum16s-202...  U7ff08586a8c5baf8600a7ce0310a8c11-4sgum16s-202...     True


##
batch = 64
output = 'variable'
generator = {
    'data' : data.generator(table=tabulation.data, batch=batch, mode='data', output=output),
    # "train" : data.generator(table=tabulation.train, batch=batch, mode='train', output=output),
    # "validation" : data.generator(table=tabulation.validation, batch=batch, mode='validation', output=output)
    # "test" : data.generator(table=tabulation.test, batch=batch, mode='test'),
}


##
generation = generator['data'].flow()
generation['binary'] = {
    "size":generation['feature'][:,1],
    "period":generation['feature'][:,2] + generation['feature'][:,5],
    "gender":generation['feature'][:,10],
    "change":generation['feature'][:,8],
    "age":generation['feature'][:,13],
    "lower":generation['target'][:,0],
    "higher":generation['target'][:,1]
}
import pandas
generation['binary'] = pandas.DataFrame(generation['binary'])
text = {
    'size':{0:"沒有", 1:"有"}, 
    'period':{0:"不記得/一年以上", 1:"一個月之內/一個月至一年"}, 
    "gender":{0:"女性/不想回答", 1:"男性"}, 
    "change":{0:"沒變化/不記得", 1:"有變化"}, 
    "age":{0:"64歲以下", 1:"65歲以上"}
}
generation['binary'] = generation['binary'].replace(text)
generation['binary'].groupby("size")[['higher','lower']].sum()
generation['binary'].groupby("period")[['higher','lower']].sum()
generation['binary'].groupby("change")[['higher','lower']].sum()
generation['binary'].groupby("age")[['higher','lower']].sum()
generation['binary'].groupby("gender")[['higher','lower']].sum()

