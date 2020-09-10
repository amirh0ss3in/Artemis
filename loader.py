import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import keras
from keras.models import Model
from keras.layers import Dense,Flatten
from keras.applications import resnet50 
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

negative=np.load("/content/drive/My Drive/DCP/DCP_positive_b.npy")
positive=np.load("/content/drive/My Drive/DCP/DCP_negative_b.npy")

n=negative[:,5:20]
p=positive[:,5:20]

nn=list()
pp=list()

data=list()
label=list()

for i in n:
  for j in i:
    nn.append(j)

for i in p:
  for j in i:
    pp.append(j)

ny= np.zeros(len(nn),dtype=int)
py= np.ones(len(pp),dtype=int)


for i in nn:
  data.append(i)
for i in pp:
  data.append(i)

for i in ny:
  label.append(i)
for i in py:
  label.append(i)

data=np.array(data)
label=np.array(label)
label=to_categorical(label)

trainX, testX, trainY, testY = train_test_split(
    data, label, test_size=0.2, random_state=42)

