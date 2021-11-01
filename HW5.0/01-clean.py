import os
import pandas as pd
import numpy as np
import shutil
from tensorflow.keras import utils

# ------------------------
# Preprocessing Data to the IMDB format
# ------------------------
print(os.getcwd())
filename = 'biology'
with open(filename + '.txt') as f:
    contents = f.read()
base_dir = os.getcwd()+'/'+filename
os.mkdir(base_dir)

print("=====from text file to chunks=====")
def get_chunks(s, maxlength):
    start = 0
    end = 0
    while start + maxlength  < len(s) and end != -1:
        end = s.rfind("\n", start, start + maxlength + 1)
        yield s[start:end]
        start = end +1
    yield s[start:]


chunks = get_chunks(contents[0:2000000], 1000)


print("=====save chunks to multiple files as the imdb format=====")
print("exclude the first 20 chunks as they are usually not the main chapters")
i =0
for n in chunks:
    if len(n) >=200:
        if i >=20 and i <1320:
            with open('./'+filename+'/'+str(i) + '.txt', "w") as f1:
                f1.write(n)
        i +=1

print("save the txt files into different dirs as the imdb data")
base_dir = os.getcwd()
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)

test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_1_dir = os.path.join(train_dir, 'bio')
os.mkdir(train_1_dir)
train_2_dir = os.path.join(train_dir, 'mc')
os.mkdir(train_2_dir)
train_3_dir = os.path.join(train_dir, 'psy')
os.mkdir(train_3_dir)

test_1_dir = os.path.join(test_dir, 'bio')
os.mkdir(test_1_dir)
test_2_dir = os.path.join(test_dir, 'mc')
os.mkdir(test_2_dir)
test_3_dir = os.path.join(test_dir, 'psy')
os.mkdir(test_3_dir)


fnames = ['{}.txt'.format(i) for i in range(20,1020)]
for fname in fnames:
    src = os.path.join(base_dir+'/biology', fname)
    dst = os.path.join(train_1_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.txt'.format(i) for i in range(1020, 1318)]
for fname in fnames:
    src = os.path.join(base_dir+'/biology', fname)
    dst = os.path.join(test_1_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.txt'.format(i) for i in range(20, 1020)]
for fname in fnames:
    src = os.path.join(base_dir + '/monte_cristo', fname)
    dst = os.path.join(train_2_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.txt'.format(i) for i in range(1020, 1318)]
for fname in fnames:
    src = os.path.join(base_dir + '/monte_cristo', fname)
    dst = os.path.join(test_2_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.txt'.format(i) for i in range(20, 1020)]
for fname in fnames:
    src = os.path.join(base_dir + '/psychology', fname)
    dst = os.path.join(train_3_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.txt'.format(i) for i in range(1020, 1318)]
for fname in fnames:
    src = os.path.join(base_dir + '/psychology', fname)
    dst = os.path.join(test_3_dir, fname)
    shutil.copyfile(src, dst)



print("=====data preprocessing =====")
#train_dir = '/Users/irene/Documents/590/HW5.0/train'
labels = []
texts = []
for label_type in ['bio', 'mc','psy']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'bio':
                labels.append('b')
            elif label_type == 'mc':
                labels.append('m')
            else:
                labels.append('p')

## as there are 3 books, encode the labels into categorical
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
dummy_y = utils.to_categorical(encoded_Y)

labels = dummy_y


print("=====saving texts and labels for further use=====")
## save texts
with open("texts.txt", "w") as f2:
    for line in texts:
        f2.write(str(line) +"\n\n\t\n")

## save labels
pd.DataFrame(labels).to_csv("labels.csv",index = None)

