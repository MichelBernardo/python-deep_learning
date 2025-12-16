import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
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


###  Linear regression  ###

# Defines the neural network parameters
linear_regression_model = keras.Sequential([
    keras.layers.Dense(1, input_dim=1, kernel_initializer='Ones', activation='linear', use_bias=False)
])
linear_regression_model.compile(loss='mean_squared_error', optimizer='adam')
print(linear_regression_model.summary())

linear_regression_model.fit(x_training, y_training)  # Training the neural network
y_training_predict = linear_regression_model.predict(x_training)  # Predicting training dataset

'''sns.lineplot(x=x_training, y=y_training, label='Training dataset')
sns.lineplot(x=x_training, y=y_training_predict[:,0], label='Training prediction')
plt.show()'''

# Viewing the data
training_predict_result = {'time':x_training, 'passageiros':y_training_predict[:,0]}
training_predict_result = pd.DataFrame(data=training_predict_result)
training_predict_result_transf = sc.inverse_transform(training_predict_result)
training_predict_result_transf = pd.DataFrame(training_predict_result_transf)
training_predict_result_transf.columns = ['time', 'passengers']

'''sns.lineplot(x='time', y='passengers', data=training_predict_result_transf, label='Training Prediction')
sns.lineplot(x='time', y='passengers', data=passengers, label='Dataset')
plt.show()'''

# Predicting Test dataset
y_test_predict = linear_regression_model.predict(x_test)

test_predict_result = {'time':x_test, 'passageiros':y_test_predict[:,0]}
test_predict_result = pd.DataFrame(data=test_predict_result)
test_predict_result_transf = sc.inverse_transform(test_predict_result)
test_predict_result_transf = pd.DataFrame(data=test_predict_result_transf)
test_predict_result_transf.columns = ['time', 'passengers']

'''sns.lineplot(x='time', y='passengers', data=passengers, label='Dataset')
sns.lineplot(x='time', y='passengers', data=test_predict_result_transf, label='Test Prediction')
sns.lineplot(x='time', y='passengers', data=training_predict_result_transf, label='Training Prediction')
plt.show()'''


###  Non-Linear Regression  ###

non_linear_regression_model = keras.Sequential([
    keras.layers.Dense(8, input_dim=1, kernel_initializer='random_uniform', activation='sigmoid', use_bias=False),
    keras.layers.Dense(8, kernel_initializer='random_uniform', activation='sigmoid', use_bias=False),
    keras.layers.Dense(1, kernel_initializer='random_uniform', activation='linear', use_bias=False)
])
non_linear_regression_model.compile(loss='mean_squared_error', optimizer='adam')
non_linear_regression_model.summary()
non_linear_regression_model.fit(x_training, y_training, epochs=500)

y_training_predict = non_linear_regression_model.predict(x_training)
y_test_predict = non_linear_regression_model.predict(x_test)

sns.lineplot(x=x_training, y=y_training, label='Training dataset')
sns.lineplot(x=x_test, y=y_test, label='Test dataset')
sns.lineplot(x=x_training, y=y_training_predict[:,0], label='Training prediction')
sns.lineplot(x=x_test, y=y_test_predict[:,0], label='Test prediction')
plt.show()