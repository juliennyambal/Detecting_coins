from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import optimizers
from keras.models import load_model
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt


model.save('coin.hdf5')
loaded_model=load_model('coin.hdf5')


data_x = np.loadtxt('Data/dataset.csv', delimiter=',')
data = np.ndarray(shape=(38,4),dtype=np.float32)


idx = np.random.permutation(len(data_x))
for i in range(len(idx)):
    data[i:,:,] = data_x[idx[i]]

X = data[:,2]
Y = data[:,-1]

label = Y.astype(np.int32)

labels = np_utils.to_categorical(label,4)

#for i in range(len(Y)):
#    print Y[i],labels[i];

model = Sequential()

model.add(Dense(3,input_dim=1, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



# Fit the model
hist = model.fit(X, labels, epochs=500, batch_size=2,validation_split=0.2)

train_loss = hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(500)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
#
plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
#plt.savefig(str(i)+'.jpg')
plt.show()

model.save_weights("coins_model.h5")
print("Saved model to disk")