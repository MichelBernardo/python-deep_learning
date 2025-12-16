import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler


###  Pre Processing  ###

# Loading the data
passengers = pd.read_csv('../data/Passengers.csv')  # import dataset

# Viewing the data
'''sns.lineplot(x='time', y='passengers', data=passengers, label='Dataset')  # Plot the dataset
mpl.rcParams['figure.figsize'] = (10, 6)  # Adjust the figure size
mpl.rcParams['font.size'] = 22  # Adjust the font size
plt.show()'''

# Normalizing the data
sc = StandardScaler()  # Declare a StandardScaler object
sc.fit(passengers)  # Calculates the dataset mean and std
normalized_data = sc.transform(passengers)  # Normalizes the data

x = normalized_data[:,0]  # Time
y = normalized_data[:,1]  # Passengers

'''sns.lineplot(x=x, y=y,label='Normalized Dataset')
plt.xlabel = 'Time'
plt.ylabel = 'Passengers'
plt.show()'''

# Splitting the dataset into training and test dataset
training_part = 0.9
training_length = int(len(passengers)*training_part)
test_length = len(passengers) - training_length

x_training = x[0:training_length]
y_training = y[0:training_length]

x_test = x[training_length:len(passengers)]
y_test = y[training_length:len(passengers)]

'''sns.lineplot(x=x_training, y=y_training, label='Training Dataset')
sns.lineplot(x=x_test, y=y_test, label='Test Dataset')
plt.xlabel = 'Time'
plt.ylabel = 'Passengers'
plt.show()'''


### Adjusting the Dataset  ###

def regression_steps(vector, n_steps):
    new_x, new_y = [], []

    for i in range(n_steps, vector.shape[0]):
        new_x.append(list(vector.loc[i-n_steps:i-1]))
        new_y.append(vector.loc[i])

    new_x, new_y = np.array(new_x), np.array(new_y)

    return new_x, new_y

steps = 1

aux_vector = pd.DataFrame(y_training)[0]
new_x_training, new_y_training = regression_steps(aux_vector, steps)

aux_vector = pd.DataFrame(y_test)[0]
new_x_test, new_y_test = regression_steps(aux_vector, steps)


###  Neural Network  ###
regression_model = keras.Sequential([
    keras.layers.Dense(8, input_dim=steps, kernel_initializer='ones', activation='linear', use_bias=False),
    keras.layers.Dense(64, kernel_initializer='random_uniform', activation='sigmoid', use_bias=False),
    keras.layers.Dense(1, kernel_initializer='random_uniform', activation='linear', use_bias=False)
])

regression_model.compile(loss='mean_squared_error', optimizer='adam')
regression_model.fit(new_x_training, new_y_training, epochs=100)

y_training_predict = regression_model.predict(new_x_training)
y_test_predict = regression_model.predict(new_x_test)

y_training_predict = pd.DataFrame(y_training_predict)[0]
y_test_predict = pd.DataFrame(y_test_predict)[0]

# sns.lineplot(x='time', y=new_y_training, data=passengers[steps:passengers.shape[1]], label='Training Dataset')
sns.lineplot(x='time', y=new_y_training, data=passengers[1:129], label='Training Dataset')
sns.lineplot(x='time', y=y_training_predict, data=passengers[1:129], label='Training Prediction')
sns.lineplot(x='time', y=new_y_test, data=passengers[130:144], label='Test Dataset')
sns.lineplot(x='time', y=y_test_predict.values, data=passengers[130:144], label='Test Prediction')
plt.xlabel('time')
plt.ylabel('passengers')
plt.show()