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
from keras.applications import resnet50 

x= np.load('data_kaggle.npy')
y= np.load('label_kaggle.npy')

print(x.shape,y.shape)



n=0
p=0
for i in y:
    if i==0:
        n+=1
    if i==1:
        p+=1
print(n,p)

y=to_categorical(y)


from sklearn.model_selection import train_test_split
data_train, data_test, labels_train, labels_test = train_test_split(x, y, test_size=0.20, random_state=42)


def xception_classifier(in_shape=(256,256,3)):    
    model = xception.Xception(weights='imagenet', include_top=False, input_shape=in_shape)
    flatten = Flatten()
    new_layer2 = Dense(2, activation='softmax', name='my_dense_2')
    inp2 = model.input
    out2 = new_layer2(flatten(model.output))
    model = Model(inp2, out2)
    model.summary(line_length=150)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5) , metrics=['accuracy']) 
    return model

# model=xception_classifier()
# history = model.fit(data_train,labels_train ,validation_data=(data_test,labels_test),batch_size=4, epochs=30 )

def resnet_classifier(in_shape=(256,256,3)):
    model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=in_shape)
    flatten = Flatten()
    new_layer2 = Dense(2, activation='softmax', name='my_dense_2')
    inp2 = model.input
    out2 = new_layer2(flatten(model.output))
    model = Model(inp2, out2)
    model.summary(line_length=150)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy']) 
    return model

# model=resnet_classifier()
# history = model.fit(data_train,labels_train ,validation_data=(data_test,labels_test),batch_size=4, epochs=30 )


y_pred = model.predict(data_test, batch_size=2, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(np.argmax(labels_test, axis=1), y_pred_bool)) 
