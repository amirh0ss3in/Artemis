import gzip
import numpy as np
import random
from sklearn.model_selection import train_test_split
import csv 
import cv2
import keras
from keras.models import Model
from keras.layers import Dense,Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from keras.utils import to_categorical
from keras.applications import resnet50 , DenseNet201 , xception
from keras.preprocessing import image
import tensorflow as tf

minimized_data = np.load(gzip.GzipFile('/content/drive/MyDrive/minimized_data.npy.gz', "r"),allow_pickle=True).item()

def count(c):
    im,p=0,0
    for i in c:
        p+=1
        for j in i:
            im+=1
    return p,im
def exp(c):
    train , test = train_test_split(c,test_size=0.2)
    xtrain , xtest=list(),list()
    for i in test:
            xtest.extend(i)
    for i in train:
            xtrain.extend(i)
    return np.array(xtrain), np.array(xtest)

print('Tabriz Negative',count(minimized_data['Tabriz']['Negative']),'\n',
'Tabriz Positive Mild',count(minimized_data['Tabriz']['Positive']['Mild']),'\n',
'Tabriz Positive Intermediate',count(minimized_data['Tabriz']['Positive']['Intermediate']),'\n',
'Tabriz Positive Severe',count(minimized_data['Tabriz']['Positive']['Severe']),'\n',
'Tehran Negative',count(minimized_data['Tehran']['Negative']),'\n',
'Tehran Positive Mild',count(minimized_data['Tehran']['Positive']['Mild']),'\n',
'Tehran Positive Intermediate',count(minimized_data['Tehran']['Positive']['Intermediate']),'\n',
'Tehran Positive Severe',count(minimized_data['Tehran']['Positive']['Severe']))

negative_train = np.append(exp(minimized_data['Tabriz']['Negative'])[0],exp(minimized_data['Tehran']['Negative'])[0],axis=0)
positive_train = np.array([*exp(minimized_data['Tehran']['Positive']['Mild'])[0],*exp(minimized_data['Tehran']['Positive']['Intermediate'])[0],*exp(minimized_data['Tehran']['Positive']['Severe'])[0],*exp(minimized_data['Tabriz']['Positive']['Mild'])[0],*exp(minimized_data['Tabriz']['Positive']['Intermediate'])[0],*exp(minimized_data['Tabriz']['Positive']['Severe'])[0]])
negative_test = np.append(exp(minimized_data['Tabriz']['Negative'])[1],exp(minimized_data['Tehran']['Negative'])[1],axis=0)
positive_test = np.array([*exp(minimized_data['Tehran']['Positive']['Mild'])[1],*exp(minimized_data['Tehran']['Positive']['Intermediate'])[1],*exp(minimized_data['Tehran']['Positive']['Severe'])[1],*exp(minimized_data['Tabriz']['Positive']['Mild'])[1],*exp(minimized_data['Tabriz']['Positive']['Intermediate'])[1],*exp(minimized_data['Tabriz']['Positive']['Severe'])[1]])

ytrain= np.append(np.zeros(len(negative_train)),np.ones(len(positive_train)))
ytest= np.append(np.zeros(len(negative_test)),np.ones(len(positive_test)))
xtrain= np.append(negative_train,positive_train,axis=0)
xtest= np.append(negative_test,positive_test,axis=0)

def shuf(a,b):
    indices = np.arange(a.shape[0])
    np.random.shuffle(indices)
    return a[indices] , b[indices]

xtrain,ytrain=shuf(xtrain,ytrain)[0],shuf(xtrain,ytrain)[1]
xtest,ytest=shuf(xtest,ytest)[0],shuf(xtest,ytest)[1]

ytrain=to_categorical(ytrain)
ytest=to_categorical(ytest)
print(xtest.shape,ytest.shape,xtrain.shape,ytrain.shape)

P=512
def eff_classifier(in_shape=(P,P,3)):    
    model = tf.keras.applications.EfficientNetB7(weights='imagenet', include_top=False, input_shape=in_shape)
    flatten = Flatten()
    new_layer2 = Dense(2, activation='softmax', name='my_dense_2')
    inp2 = model.input
    out2 = new_layer2(flatten(model.output))
    model = Model(inp2, out2)
    # model.summary(line_length=150)
    return model


def xception_classifier(in_shape=(P,P,3)):    
    model = xception.Xception(weights='imagenet', include_top=False, input_shape=in_shape)
    flatten = Flatten()
    new_layer2 = Dense(2, activation='softmax', name='my_dense_2')
    inp2 = model.input
    out2 = new_layer2(flatten(model.output))
    model = Model(inp2, out2)
    # model.summary(line_length=150)
    return model


def resnet_classifier(in_shape=(P,P,3)):
    model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=in_shape)
    flatten = Flatten()
    new_layer2 = Dense(2, activation='softmax', name='my_dense_2')
    inp2 = model.input
    out2 = new_layer2(flatten(model.output))
    model = Model(inp2, out2)
    # model.summary(line_length=150)
    return model

aug = image.ImageDataGenerator(rotation_range=50, zoom_range=0.7,
	horizontal_flip=True)

BS=2
EPOCHS= 50
model=eff_classifier()
lr=1e-6
b1=0.8

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr,beta_1=b1) , metrics=['accuracy',tf.keras.metrics.AUC(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall()]) 
# history = model.fit( xtrain , ytrain ,validation_data=(xtest,ytest) ,batch_size=BS, epochs=EPOCHS )

history = model.fit_generator(aug.flow(xtrain, ytrain, batch_size=BS ) ,
                              validation_data=(xtest, ytest), steps_per_epoch=len(xtrain) // BS,
                              epochs=EPOCHS)

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
