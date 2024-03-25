import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import StandardScaler


###  Pre processing  ###

# Loading Data
bicycles = pd.read_csv('bicicletas.csv')
bicycles['datas'] = pd.to_datetime(bicycles['datas'])

'''# Viewing the dataset
sns.lineplot(x='datas', y='contagem', data=bicycles)
plt.xticks(rotation=90)
plt.show()'''

# Normalizing the data
sc = StandardScaler()
sc.fit(bicycles['contagem'].values.reshape(-1,1))
y = sc.transform(bicycles['contagem'].values.reshape(-1,1))

# Spliting the dataset into training and test set
training_size = int(len(bicycles)*0.9) 
test_size = len(bicycles) - training_size

y_training = y[0:training_size]
y_test = y[training_size:len(bicycles)]

'''sns.lineplot(x='datas', y=y_training[:,0], data=bicycles[0:training_size])
sns.lineplot(x='datas', y=y_test[:,0], data=bicycles[training_size:len(bicycles)])
plt.xticks(rotation=90)
plt.show()'''


### Adjusting the Dataset  ###

def regression_steps(vector, n_steps):
    new_x, new_y = [], []

    for i in range(n_steps, vector.shape[0]):
        new_x.append(list(vector.loc[i-n_steps:i-1]))
        new_y.append(vector.loc[i])

    new_x, new_y = np.array(new_x), np.array(new_y)

    return new_x, new_y

steps = 10

aux_vector = pd.DataFrame(y_training)[0]
new_x_training, new_y_training = regression_steps(aux_vector, steps)

aux_vector = pd.DataFrame(y_test)[0]
new_x_test, new_y_test = regression_steps(aux_vector, steps)

###  LSTM  ###
new_x_training = new_x_training.reshape((new_x_training.shape[0], new_x_training.shape[1], 1))
new_x_test = new_x_test.reshape((new_x_test.shape[0], new_x_test.shape[1], 1))

recurrent = keras.Sequential([
    keras.layers.LSTM(128, input_shape=(new_x_training.shape[1], new_x_training.shape[2])),
    keras.layers.Dense(units=1)
])

recurrent.compile(loss='mean_squared_error', optimizer='RMSProp')
recurrent.summary()

result = recurrent.fit(new_x_training, new_y_training, validation_data=(new_x_test, new_y_test), epochs=100)

y_training_predict = recurrent.predict(new_x_training)

sns.lineplot(x='datas', y=y_training[:,0], data=bicycles[0:training_size], label='Training Dataset')
sns.lineplot(x='datas', y=y_training_predict[:,0], data=bicycles[0:15662], label='Training Prediction')
plt.xticks(rotation=70)
plt.show()

y_test_prediction = recurrent.predict(new_x_test)

sns.lineplot(x='datas', y=y_test[:,0], data=bicycles[training_size:len(bicycles)], label='Test Dataset')
sns.lineplot(x='datas', y=y_test_prediction[:,0], data=bicycles[training_size+steps:len(bicycles)], marker='.', label='Test Prediction')
plt.xticks(rotation=70)
plt.plot()