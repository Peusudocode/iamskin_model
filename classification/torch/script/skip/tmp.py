import pandas
import os
table = pandas.read_csv("resource/20211231v2/csv/information.csv")
image = os.listdir('resource/20211231v2/jpg/crop/lower/type-a') + os.listdir('resource/20211231v2/jpg/crop/higher/type-a')
label = ['resource/20211231v2/jpg/crop/lower/type-a', 'resource/20211231v2/jpg/crop/higher/type-a']
exist = []
for index, item in table.iterrows():

    if(item['image_crop'] in image): exist += ['exist']
    if(item['image_crop'] not in image): exist += ['not exist']    
    pass

table['status'] = exist
table = table.loc[table['status']=='exist'].reset_index(drop=True)
table.to_csv("resource/20211231v2/csv/information_.csv", index=False)
