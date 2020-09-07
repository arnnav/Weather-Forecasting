

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


data = pd.read_csv("Dataset_Final.csv").set_index('DATE')
data.info()




target = "COND"
features = ["TEMP","DEWP","SLP","VISIB","WDSP","PRCP"]

train, test = train_test_split(data, test_size=0.1)

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

print("Dimensions of the training set : {0}".format(np.shape(X_train)))
print("Dimensions of the training set (target) : {0}".format(np.shape(y_train.values.reshape(len(y_train),1))))


# In[4]:


def model(hu, model_dir, features):
    feature_columns = [tf.feature_column.numeric_column("x", shape=[len(features),1])]

    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                        hidden_units=hu,
                                        n_classes=3,                                    
                                        model_dir=model_dir)
    # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(X_train)},
        y=np.array(y_train.values.reshape((len(y_train),1))),
        num_epochs=None,
        shuffle=True,
        batch_size=8000)
    return classifier, train_input_fn


classifier, train_input_fn = model([50,50,50,50,50], "./DNN", features)
classifier.train(input_fn=train_input_fn, steps=40000)


def testinput(X_test, y_test):
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(X_test)},
        y=np.array(y_test),
        num_epochs=1,
        shuffle=False)
    return test_input_fn


# Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=testinput(X_test,y_test))["accuracy"]
print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
my_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(X_test[features])},
        y=None,
        num_epochs=1,
        shuffle=False)
pred = classifier.predict(input_fn=my_input_fn)



predictions = list(pred)
predictions[0]
print (predictions[0])




final_pred = np.array([])
for p in predictions:
    final_pred = np.append(final_pred,p['class_ids'][0])
final_pred = final_pred.astype(int)
