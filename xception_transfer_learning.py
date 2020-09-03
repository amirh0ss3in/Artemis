import keras
from keras.models import Model
from keras.layers import Dense,Flatten
from keras.applications import xception 
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

trainX=np.load("C:/Users/Amirhossein/Desktop/trainX.npy")
testX=np.load("C:/Users/Amirhossein/Desktop/testX.npy")
trainY=np.load("C:/Users/Amirhossein/Desktop/trainY.npy")
testY=np.load("C:/Users/Amirhossein/Desktop/testY.npy")

print(trainX.shape,testX.shape)

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

model=xception_classifier()
history = model.fit(trainX,trainY , validation_data=(testX,testY ), batch_size=15, epochs=100 )

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

y_pred = model.predict(testX, batch_size=2, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(np.argmax(testY, axis=1), y_pred_bool))
