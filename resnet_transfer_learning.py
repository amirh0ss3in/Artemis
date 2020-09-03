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

n=negative[:,10:15]
p=positive[:,10:15]

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



print(trainX.shape,testX.shape)

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

model=resnet_classifier()
history = model.fit(trainX,trainY , validation_data=(testX,testY), batch_size=15, epochs=100 )

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig('accuracy_history_resnet')
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig('loss_history_resnet')
plt.close()

y_pred = model.predict(testX, batch_size=2, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(np.argmax(testY, axis=1), y_pred_bool))
