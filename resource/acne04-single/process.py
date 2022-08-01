
##  重點是根據座標資訊的 xml 檔案，將單一個物件從影像中擷取出來，
##  依據類別進行存放。

##
import os
import shutil
import PIL.Image
import xml.etree.ElementTree
import tqdm
import re
import sklearn.model_selection

##
folder = 'resource/acne04-default/'
nest = os.path.join(folder, 'xml')
loop = os.listdir(nest)
for i in  tqdm.tqdm(loop): 
    
    # i = loop[0]
    tree = xml.etree.ElementTree.parse(os.path.join(nest, i))
    root = tree.getroot()
    name = root.find('filename').text
    level = name.split("_")[0]
    image = os.path.join(folder, "jpg", name)
    image = PIL.Image.open(image)
    
    for index, o in enumerate(root.findall('object')):

        # o = root.findall('object')[0]
        position = [
        int(o.find('bndbox').find('xmin').text),
        int(o.find('bndbox').find('ymin').text),
        int(o.find('bndbox').find('xmax').text),
        int(o.find('bndbox').find('ymax').text)
        ]
        destination = 'resource/acne04-single/jpg/{}'.format(level)
        os.makedirs(destination, exist_ok=True)
        link = os.path.join(destination, name.replace('.jpg', "-{}.jpg".format(index)))
        image.crop(position).save(link)
        continue
    
    continue

##
nest = 'resource/acne04-single/jpg/'
level = os.listdir(nest)
data = []
for l in level:

    h = os.path.join(nest, l)
    data += [os.path.join(h, t).replace('\\', '/') for t in os.listdir(h)]
    continue

data = sorted(data)

##
train, group = sklearn.model_selection.train_test_split(data, test_size=0.2, random_state=0)
validation, test = sklearn.model_selection.train_test_split(group, test_size=0.5, random_state=0)
train[:6]
validation[:6]
test[:6]
pair = zip(['train', "validation", 'test'], [train, validation, test])
for mode, loop in pair:
    
    head = os.path.join('resource/acne04-single', mode)
    os.makedirs(head, exist_ok=True)
    for i in tqdm.tqdm(loop):
        
        name = re.split('/', i)[-1]
        level = re.split('/', i)[-2]
        body = os.path.join(head, level).replace('\\', '/')
        os.makedirs(body, exist_ok=True)
        foot = os.path.join(body, name).replace('\\', '/')
        _ = shutil.copy(i, foot)
        continue
    
    continue


