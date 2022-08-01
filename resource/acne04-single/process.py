
##  重點是根據座標資訊的 xml 檔案，將單一個物件從影像中擷取出來，
##  依據類別進行存放。

##
import os
import shutil
import numpy
import pandas
import PIL.Image
import xml.etree.ElementTree
import tqdm

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
