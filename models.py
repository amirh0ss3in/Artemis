import numpy as np
import csv 
import cv2

CT_NonCOVID=np.load('/content/drive/My Drive/dataset/CT_NonCOVID.npy',allow_pickle=True)
CT_COVID= np.load('/content/drive/My Drive/dataset/CT_COVID.npy',allow_pickle=True)

P=480

def read(s):
    d=list()
    with open(s,'r') as csvfile: 
        reader = csv.reader(csvfile, delimiter=',', quotechar='|') 
        for row in reader:
            d.append(row[0])
    return np.array(d)

def split(d,s):
    data=list()
    for i in d:
        if i[1] in read(s):
            data.append(i[0])
    return np.array(data)

def rs(a):
    x=list()
    for i in a:    
        res = cv2.resize(i , dsize=(P,P), interpolation=cv2.INTER_CUBIC)
        x.append(res)
    x=np.array(x)
    return x

def combine(positive,negative):
    data , label=list() , list()
    for i in range(len(positive)):
        label.append(1)
    for i in range(len(negative)):
        label.append(0)
    for i in positive:
        data.append(i)
    for i in negative:
        data.append(i)
    return np.array(data),np.array(label)


# print('trainCT_COVID:',rs(split(CT_COVID,'trainCT_COVID.csv')).shape)
# print('testCT_COVID:',rs(split(CT_COVID,'testCT_COVID.csv')).shape)
# print('valCT_COVID:',rs(split(CT_COVID,'valCT_COVID.csv')).shape)
# print('trainCT_NonCOVID:',rs(split(CT_NonCOVID,'trainCT_NonCOVID.csv')).shape)
# print('testCT_NonCOVID:',rs(split(CT_NonCOVID,'testCT_NonCOVID.csv')).shape)
# print('valCT_NonCOVID:',rs(split(CT_NonCOVID,'valCT_NonCOVID.csv')).shape)


# train=combine(rs(split(CT_COVID,'trainCT_COVID.csv')),rs(split(CT_NonCOVID,'trainCT_NonCOVID.csv')))
# test=combine(rs(split(CT_COVID,'testCT_COVID.csv')),rs(split(CT_NonCOVID,'testCT_NonCOVID.csv')))
# val= combine(rs(split(CT_COVID,'valCT_COVID.csv')),rs(split(CT_NonCOVID,'valCT_NonCOVID.csv')))

def join(input1,input2):
    data=list()
    label=list()
    for i in input1[0]:
        data.append(i)
    for i in input2[0]:
        data.append(i)
    for i in input1[1]:
        label.append(i)
    for i in input2[1]:
        label.append(i)
    return np.array(data) , np.array(label)
    

import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense,Flatten
from keras.applications import xception 
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from keras.utils import to_categorical
import cv2
from keras.applications import resnet50 , DenseNet201 
from keras.preprocessing import image
import tensorflow as tf

train=combine(rs(split(CT_COVID,'/content/drive/My Drive/dataset/trainCT_COVID.csv')),rs(split(CT_NonCOVID,'/content/drive/My Drive/dataset/trainCT_NonCOVID.csv')))
test=combine(rs(split(CT_COVID,'/content/drive/My Drive/dataset/testCT_COVID.csv')),rs(split(CT_NonCOVID,'/content/drive/My Drive/dataset/testCT_NonCOVID.csv')))
val= combine(rs(split(CT_COVID,'/content/drive/My Drive/dataset/valCT_COVID.csv')),rs(split(CT_NonCOVID,'/content/drive/My Drive/dataset/valCT_NonCOVID.csv')))

aug = image.ImageDataGenerator(rotation_range=10, zoom_range=0.5,
	horizontal_flip=True)

def xception_classifier(in_shape=(P,P,3)):    
    model = xception.Xception(weights='imagenet', include_top=False, input_shape=in_shape)
    flatten = Flatten()
    new_layer2 = Dense(2, activation='softmax', name='my_dense_2')
    inp2 = model.input
    out2 = new_layer2(flatten(model.output))
    model = Model(inp2, out2)
    model.summary(line_length=150)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00001 , beta_1=0.5) , metrics=['accuracy',tf.keras.metrics.AUC(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall()]) 
    return model


def resnet_classifier(in_shape=(P,P,3)):
    model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=in_shape)
    flatten = Flatten()
    new_layer2 = Dense(2, activation='softmax', name='my_dense_2')
    inp2 = model.input
    out2 = new_layer2(flatten(model.output))
    model = Model(inp2, out2)
    model.summary(line_length=150)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001) , metrics=['accuracy',tf.keras.metrics.AUC(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall()]) 
    return model

BS=4
EPOCHS=10
model=xception_classifier()

# history = model.fit_generator(aug.flow(join(train,val)[0], to_categorical(join(train,val)[1]), batch_size=BS ) ,
# 	validation_data=(test[0], to_categorical(test[1])), steps_per_epoch=len(train[0]) // BS,
# 	epochs=EPOCHS)

history = model.fit( join(train,val)[0] , to_categorical(join(train,val)[1]) ,validation_data=(test[0],to_categorical(test[1])) ,batch_size=BS, epochs=EPOCHS )

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig('accuracy_history_xception')
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig('loss_history_xception')
plt.close()

y_pred = model.predict(test[0], batch_size=2, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(test[1], y_pred_bool)) 



