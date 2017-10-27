import sys
import csv
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# readin
# (32561, 106)
x_train = pandas.read_csv(sys.argv[1]).values.astype(numpy.float)
# (32561, 1)
y_train = pandas.read_csv(sys.argv[2]).values.astype(numpy.float)
# (16281, 106)
x_test = pandas.read_csv(sys.argv[3]).values.astype(numpy.float)
print(x_test.shape)
# 0~1
for i in range(x_train.shape[1]):
    x_train[:,i] = (x_train[:,i] - min(x_train[:,i])) / (max(x_train[:,i]) - min(x_train[:,i]))
for i in range(x_test.shape[1]):
    if (max(x_test[:,i]) - min(x_test[:,i])) != 0:
        x_test[:,i] = (x_test[:,i] - min(x_test[:,i])) / (max(x_test[:,i]) - min(x_test[:,i]))

model = Sequential()
model.add(Dense(1024, input_dim=106, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=10,
          batch_size=128)
y = model.predict(x_test, batch_size=128)
print(y)

with open(sys.argv[4], 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['id', 'label'])
    for i in range(y.shape[0]):
        if y[i] > 0.5:
            spamwriter.writerow([ str(i+1) , '1' ])
        else:
            spamwriter.writerow([ str(i+1) , '0' ])
