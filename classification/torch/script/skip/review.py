

##
import data
import pandas


##
tabulation = data.tabulation()
tabulation.load(train='resource/csv/1101/train.csv', validation='resource/csv/1101/validation.csv')
tabulation.train['prediction'].replace({'lower':0, "higher":1}, inplace=True)
tabulation.validation['prediction'].replace({'lower':0, "higher":1}, inplace=True)
tabulation.train['target'] = tabulation.train['vote'].replace({'lower':0, "higher":1})
tabulation.validation['target'] = tabulation.validation['vote'].replace({'lower':0, "higher":1})
total = pandas.concat([tabulation.train, tabulation.validation], axis=0)


##
print("-"*100)
print("total {} ...".format(len(total)))
print(data.summary(table=total, label=[1,0]), '\n')
print(total['vote'].value_counts())
print(total['mole_size'].value_counts())
print(total['change'].value_counts())
print(total['gender'].value_counts())
print(total['age'].value_counts())
print(total['period'].value_counts())
print("-"*100)
print("train data {} ...".format(len(tabulation.train)))
print(data.summary(table=tabulation.train, label=[1,0]), '\n')
print(tabulation.train['vote'].value_counts())
print(tabulation.train['mole_size'].value_counts())
print(tabulation.train['change'].value_counts())
print(tabulation.train['gender'].value_counts())
print(tabulation.train['age'].value_counts())
print(tabulation.train['period'].value_counts())
print("-"*100)
print("validation data {} ...".format(len(tabulation.validation)))
print(data.summary(table=tabulation.validation, label=[1,0]), '\n')
print(tabulation.validation['vote'].value_counts())
print(tabulation.validation['mole_size'].value_counts())
print(tabulation.validation['change'].value_counts())
print(tabulation.validation['gender'].value_counts())
print(tabulation.validation['age'].value_counts())
print(tabulation.validation['period'].value_counts())
print("-"*100)




# print("total data")
# len(tabulation.data)
# tabulation.data['vote'].value_counts()
# print(data.summary(table=tabulation.data, label=[1,0]))
# print("\n")


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
##

# tabulation.train['vote'].value_counts()
# print("\n")
# print("validation data")
# print(data.summary(table=tabulation.validation, label=[1,0]))
# len(tabulation.validation)
# tabulation.validation['vote'].value_counts()
# print("\n")


# ##
# batch = 64
# output = 'image and variable'
# generator = {
#     'data' : data.generator(table=tabulation.train, batch=batch, mode='data', output=output),
#     "train" : data.generator(table=tabulation.train, batch=batch, mode='train', output=output),
#     "validation" : data.generator(table=tabulation.validation, batch=batch, mode='validation', output=output)
#     # "test" : data.generator(table=tabulation.test, batch=batch, mode='test'),
# }


# ##
# '''
# 現有模型
# '''