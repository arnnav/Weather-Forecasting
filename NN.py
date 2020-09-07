
import csv
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import explained_variance_score,     mean_absolute_error,     median_absolute_error
from sklearn.model_selection import train_test_split



import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

df = pd.read_csv('Dataset_Final.csv').set_index('DATE')
df.describe().T
df.info()




df = df.drop(['COND'], axis=1)


# X will be a pandas dataframe of all columns except temp
X = df[[col for col in df.columns if col != 'TEMP']]

# y will be a pandas series of the temp
y = df['TEMP']

# split data into training set and a temporary set using sklearn.model_selection.traing_test_split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=23)

# take the remaining 20% of data in X_tmp, y_tmp and split them evenly
X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=23)


X_train.shape, X_test.shape, X_val.shape
print("Training instances   {}, Training features   {}".format(X_train.shape[0], X_train.shape[1]))
print("Validation instances {}, Validation features {}".format(X_val.shape[0], X_val.shape[1]))
print("Testing instances    {}, Testing features    {}".format(X_test.shape[0], X_test.shape[1]))




feature_cols = [tf.feature_column.numeric_column(col) for col in X.columns]

regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                      hidden_units=[50, 50, 50, 50, 50],
									  activation_fn=tf.nn.relu,
									  optimizer='Adagrad',
                                      model_dir='tf_wx_model')





def wx_input_fn(X, y=None, num_epochs=None, shuffle=True, batch_size=7000):
    return tf.estimator.inputs.pandas_input_fn(x=X,
                                               y=y,
                                               num_epochs=num_epochs,
                                               shuffle=shuffle,
                                               batch_size=batch_size)

evaluations = []
STEPS = 400
for i in range(100):
    regressor.train(input_fn=wx_input_fn(X_train, y=y_train), steps=STEPS)
    evaluation = regressor.evaluate(input_fn=wx_input_fn(X_val, y_val,
                                                         num_epochs=1,
                                                         shuffle=False),
                                    steps=1)
    evaluations.append(regressor.evaluate(input_fn=wx_input_fn(X_val,
                                                               y_val,
                                                               num_epochs=1,
                                                               shuffle=False)))


evaluations[0]



import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [14, 10]

loss_values = [ev['loss'] for ev in evaluations]
training_steps = [ev['global_step'] for ev in evaluations]

plt.scatter(x=training_steps, y=loss_values)
plt.xlabel('Training steps (Epochs = steps / 2)')
plt.ylabel('Loss (SSE)')
plt.show()



pred = regressor.predict(input_fn=wx_input_fn(X_test,
                                              num_epochs=1,
                                              shuffle=False))






											  
											  
											  

predictions = np.array([p['predictions'][0] for p in pred])


#list(predictions)
print (predictions)




test=pd.DataFrame()
test['DATE']=y_test.index
test['TEMP']=predictions
test.to_csv('Test.csv', sep=',', encoding='utf-8')

xtest=pd.DataFrame()
xtest['DATE']=X_test.index
xtest['TEMP']=y_test[0]
xtest.to_csv('y_test.csv', sep=',', encoding='utf-8')



print("The Mean Absolute Error: %.2f degrees Celcius" % mean_absolute_error(
                                            y_test, predictions))  
print("The Median Absolute Error: %.2f degrees Celcius" % median_absolute_error(
                                            y_test, predictions))
print("The Median Absolute Error: %.2f degrees Celcius" % median_absolute_error(
                                            y_test, predictions))


