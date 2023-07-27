import tensorflow as tf
from tensorflow.keras import utils
import pandas as pd
from app.arch import get_model
from app.input_function import input_fun

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = utils.get_file('iris_training.csv',
                            "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = utils.get_file('iris_test.csv', "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train_data = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test_data = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

# pop out the species column from both
train_y = train_data.pop('Species')
test_y = test_data.pop('Species')

feature_columns = []
for key in train_data.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))

# Get the model
classifier = get_model(feature_columns=feature_columns)

# Training
classifier.train(
    input_fn=lambda: input_fun(train_data, train_y, training=True),
    steps=5000
)

# Evaluation
result = classifier.evaluate(input_fn=lambda: input_fun(test_data, test_y, training=False))
print(f"\nTest set accuracy: {result['accuracy']}")


# test
def test_input_func(features, batch_size=256):
    # Convert the inputs to a dataset without labele
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)


input_features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted.")
for feature in input_features:
    valid = True
    while valid:
        value = input(feature + ": ")
        if not value.isdigit():
            valid = False
    predict[feature] = [float(value)]

predictions = classifier.predict(input_fn=lambda: test_input_func(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print(f'Prediction is {SPECIES[class_id]} ({100*probability})')