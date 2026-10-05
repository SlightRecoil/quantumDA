# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the wine quality dataset and print sample data and describe the data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
# Pandas already supplies a function for reading CSV files
# Hint: https://pandas.pydata.org/docs/user_guide/10min.html

# TODO: read the dataset with pandas and assign it to raw_dataset
raw_dataset = pd.read_csv(url, sep=";")
raw_dataset.head()

# Separate the dataset into features and labels
# TODO: Try it yourself (hint: remove the column with the labels)
features = dataset.drop("quality", axis=1)
labels = dataset["quality"]

# Implementing a preprocessing method which can be used in more projects
# Learn more about MinMax and Standard scaling
# https://stackoverflow.com/questions/62178888/can-someone-explain-to-me-how-minmaxscaler-works
# https://stackoverflow.com/questions/40758562/can-anyone-explain-me-standardscaler

# Sklearn already implements these two scaler functions
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def normalize_data(data, method=None):
  """ This method returns normalized data depending on the selected method
      None means no normalization (no processing done)
      "MinMax" means with sklearn MinMaxSclaer
      "Std" means with sklearn StdScaler """
  if method == "MinMax":
    scaler = MinMaxScaler()
  elif method == "Std":
    scaler = StandardScaler()
  else:
    return data                 # No scaling or normalization done

  # Apply scaling/normalization according to selected method and return the data
  scaler.fit(data)
  return scaler.transform(data)

# Set random seed
tf.random.set_seed(42)

# Define the model, show the summary
# Let's try a model with 20, 20, 10, 1 neurons (in a total of 4 layers)
model = keras.Sequential([
    layers.Input((11,)),
    layers.Dense(20, activation="relu"),
    layers.Dense(20, activation="relu"),
    layers.Dense(10, activation="relu"),
    layers.Dense(1)])

# Implementation of early stopping and model checkpoint
# Callbacks which we will use in many projects

from keras.callbacks import EarlyStopping, ModelCheckpoint
# callbacks for model.fit
# early stopping of training if the accuracy doesn't get better after patience epochs
es = EarlyStopping(monitor='loss', mode='min', patience=20,  restore_best_weights=True)

# ModelCheckpoint Callback is used to save the model periodically to files.
# If save_best_only is True then it only saves when the val_accuracy is getting better.
filepath="weights-improvement-{epoch:02d}.keras"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='max')
callbacks_list = [es]

# Train the model for 100 epochs
chat# TODO: Try it yourself
model.fit(X_train, Y_train, epochs=100, batch_size=100)

Y_pred = model.predict(X_test)

# Visualize the actual vs. predicted values
plt.scatter(Y_test, Y_pred)
plt.xlabel('True Quality')
plt.ylabel('Predicted Quality')
plt.title('Wine Quality: Actual vs. Predicted')
plt.show()