import tensorflow as tf
from tensorflow.keras import utils
import pandas as pd


CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = utils.get_file('iris_training.csv', "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = utils.get_file('iris_test.csv', "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train_data = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test_data = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

# pop out the species column from both
train_y = train_data.pop('Species')
test_y = test_data.pop('Species')

print(train_data.head())