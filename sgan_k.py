import numpy as np
import csv 
import cv2

CT_NonCOVID=np.load('/content/drive/My Drive/dataset/CT_NonCOVID.npy',allow_pickle=True)
CT_COVID= np.load('/content/drive/My Drive/dataset/CT_COVID.npy',allow_pickle=True)

P=256

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
from sklearn.model_selection import train_test_split
 
from keras import backend
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, Lambda, Activation
import numpy as np
from numpy import zeros, ones, asarray
from numpy.random import randn , randint
import cv2
from sklearn.metrics import classification_report
from matplotlib import pyplot
from keras.utils import to_categorical


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



trainX, testX, trainY, testY = join(train,val)[0],test[0],join(train,val)[1],test[1]



# resize the data
xtr=list()
for i in trainX:    
    res = cv2.resize(i , dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    res=np.dot(res, [0.299, 0.587, 0.114])
    res=np.reshape(res,[128, 128,1])
    xtr.append(res)
xtr=np.array(xtr)

xte=list()
for i in test[0]:    
    res = cv2.resize(i , dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    res=np.dot(res, [0.299, 0.587, 0.114])
    res=np.reshape(res,[128, 128,1])
    xte.append(res)
xte=np.array(xte)

# scale data to [-1,1]
xte= (xte-(np.max(xte)+np.min(xte))/2)/(-(np.max(xte)+np.min(xte))/2)
xte/=np.max(xte)
xtr= (xtr-(np.max(xtr)+np.min(xtr))/2)/(-(np.max(xtr)+np.min(xtr))/2)
xtr/=np.max(xtr)

# custom activation function
def custom_activation(output):
    logexpsum = backend.sum(backend.exp(output), axis=-1, keepdims=True)
    result = logexpsum / (logexpsum + 1.0)
    return result

# define the standalone supervised and unsupervised discriminator models
def define_discriminator(in_shape=(128,128,1), n_classes=2):
    # image input
    in_image = Input(shape=in_shape)
    # downsample
    x = Conv2D(128, (5,5), strides=(1,1), padding='same')(in_image)
    x = LeakyReLU(alpha=0.2)(x)
    # downsample
    x = Conv2D(128, (5,5), strides=(2,2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    # downsample
    x = Conv2D(128, (5,5), strides=(2,2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    # flatten feature maps
    x = Flatten()(x)
    # dropout
    # x = Dropout(0.4)(x)
    # output layer nodes
    x = Dense(n_classes)(x)
    # supervised output
    c_out_layer = Activation('softmax')(x)
    # define and compile supervised discriminator model
    c_model = Model(in_image, c_out_layer)
    c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.00002, beta_1=0.5), metrics=['accuracy'])
    # unsupervised output
    d_out_layer = Lambda(custom_activation)(x)
    # define and compile unsupervised discriminator model
    d_model = Model(in_image, d_out_layer)
    d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00002, beta_1=0.7))
    return d_model, c_model

# define the standalone generator model
def define_generator(latent_dim):
    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundation for 14x14 image
    n_nodes = 1024* 32 * 32
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((32, 32, 1024))(gen)
    # upsample to 28x28
    gen = Conv2DTranspose(512, (8,8), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.4)(gen)
    # upsample to 56x56
    gen = Conv2DTranspose(512, (8,8), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.4)(gen)
    # output
    out_layer = Conv2D(1, (32,32), activation='tanh', padding='same')(gen)
    # define model
    model = Model(in_lat, out_layer)
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect image output from generator as input to discriminator
    gan_output = d_model(g_model.output)
    # define gan model as taking noise and outputting a classification
    model = Model(g_model.input, gan_output)
    # compile model
    opt = Adam(lr=0.00001, beta_1=0.7)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

# load the images
def load_real_samples():
    # load dataset
    print(xtr.shape, trainY.shape)
    return [xtr, trainY]

# select a supervised subset of the dataset, ensures classes are balanced
def select_supervised_samples(dataset, n_samples=len(xtr), n_classes=2):
    X, y = dataset
    X_list, y_list = list(), list()
    n_per_class = int(n_samples / n_classes)
    for i in range(n_classes):
        # get all images for this class
        X_with_class = X[y == i]
        # choose random instances
        ix = randint(0, len(X_with_class), n_per_class)
        # add to list
        [X_list.append(X_with_class[j]) for j in ix]
        [y_list.append(i) for j in ix]
    return asarray(X_list), asarray(y_list)

# select real samples
def generate_real_samples(dataset, n_samples):
    # split into images and labels
    images, labels = dataset
    # choose random instances
    ix = randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], labels[ix]
    # generate class labels
    y = ones((n_samples, 1))
    return [X, labels], y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    z_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = z_input.reshape(n_samples, latent_dim)
    return z_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict(z_input)
    # create class labels
    y = zeros((n_samples, 1))
    return images, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, c_model, latent_dim, dataset, n_samples=100):
    # prepare fake examples
    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # plot images
    for i in range(100):
    #    # define subplot
        pyplot.subplot(10, 10, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
    # save plot to file
    filename1 = 'generated_plot_%04d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # evaluate the classifier model
    X, y = dataset
    # _, acc = c_model.evaluate(X, y, verbose=0)
    # print('Classifier training Accuracy: %.3f%%' % (acc * 100))
    # #classification report of training:
    # print(classification_report(y,np.argmax(c_model.predict(X, batch_size=2, verbose=1), axis=1)),'\n')
    _, acct = c_model.evaluate(xte, testY, verbose=0)
    print('Classifier validation Accuracy: %.3f%%' % (acct * 100))
    #classification report of validation test:
    # print(classification_report(testY, np.argmax(c_model.predict(xte, batch_size=2, verbose=1), axis=1)),'\n')


# train the generator and discriminator
def train(g_model, d_model, c_model, gan_model, dataset, latent_dim, n_epochs=20, n_batch=2,l_d=list(),l_g=list(),l_c=list()):
    # select supervised dataset
    X_sup, y_sup = select_supervised_samples(dataset)
    print(X_sup.shape, y_sup.shape)
    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs, n_batch, half_batch, bat_per_epo, n_steps))
    # manually enumerate epochs
    for i in range(n_steps):
        # update supervised discriminator (c)
        [Xsup_real, ysup_real], _ = generate_real_samples([X_sup, y_sup], half_batch)
        c_loss, c_acc = c_model.train_on_batch(Xsup_real, ysup_real)
        # update unsupervised discriminator (d)
        [X_real, _], y_real = generate_real_samples(dataset, half_batch)
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)
        # update generator (g)
        X_gan, y_gan = generate_latent_points(latent_dim, n_batch), ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        # summarize loss on this batch
        print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss))
        # plot loss over each step
        # evaluate the model performance every so often
        if (i+1) % (bat_per_epo * 1) == 0:
            summarize_performance(i, g_model, c_model, latent_dim, dataset)

# size of the latent space
latent_dim = 100
# create the discriminator models
d_model, c_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = load_real_samples()
# train model
train(g_model, d_model, c_model, gan_model, dataset, latent_dim)
