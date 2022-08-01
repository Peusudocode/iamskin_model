
##  Follow the original dataset, store in [acne04-origin] folder, 
##  transform the dataset structure without change any 
##  information of dataset.
##  重點是將資料結構做一個整理，名稱跟無異議的訊息去除。

##
import os
import shutil
import numpy
import pandas
import PIL.Image
import xml.etree.ElementTree

##
folder = 'resource/acne04-origin/'
nest = os.path.join(folder, 'Classification/JPEGImages/')
loop = os.listdir(nest)
for i in loop:

    name = i.replace('levle', 'level')
    image = PIL.Image.open(os.path.join(nest, i))
    destination = 'resource/acne04-default/jpg/'
    os.makedirs(destination, exist_ok=True)
    image.save(os.path.join(destination, name))
    continue

##
nest = os.path.join(folder, 'Classification/')
destination = 'resource/acne04-default/'
loop = [i for i in os.listdir(nest) if('txt' in i)]
for i in loop: 
    
    shutil.copyfile(os.path.join(nest, i), os.path.join(destination, i))
    continue

for i in loop:

    item = pandas.read_csv(os.path.join(destination, i), sep='\s+', header=None)
    item[0] = item[0].str.replace("levle", 'level')
    item = numpy.array(item)
    numpy.savetxt(os.path.join(destination, i), item, delimiter='  ', fmt='%s')
    continue

##
nest = os.path.join(folder, 'Detection/VOC2007/Annotations/')
loop = os.listdir(nest)
for i in  loop: 
    
    tree = xml.etree.ElementTree.parse(os.path.join(nest, i))
    root = tree.getroot()
    for index, j in enumerate(root.iter()): 
        
        if(index==0): continue
        if(j.tag=='folder'): j.text = "NULL"
        if(j.tag=='filename'): j.text = j.text.replace('levle', 'level')
        if(j.tag=='database'): j.text = "NULL"
        if(j.tag=='annotation'): j.text = "NULL"
        if(j.tag=='image'): j.text = "NULL"
        if(j.tag=='flickrid'): j.text = "NULL"
        if(j.tag=='name'): j.text = "acne"
        if(j.tag=='pose'): j.text = "NULL"
        if(j.tag=='truncated'): j.text = "NULL"
        if(j.tag=='difficult'): j.text = "NULL"
        continue
    
    destination = 'resource/acne04-default/'
    destination = os.path.join(destination, 'xml')
    os.makedirs(destination, exist_ok=True)
    name = os.path.join(destination, i.replace('levle', 'level'))
    tree.write(name)
    continue
        


