import tensorflow
import pandas


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tensorflow.data.Dataset.from_tensor_slices((dict(features),
                                                          labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tensorflow.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


def main():
    SPECIES = ['Setosa', 'Versicolor', 'Virginica']
    CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                        'PetalLength', 'PetalWidth', 'Species']

    train_path = 'traindata.csv'
    test_path = 'testdata.csv'

    train = pandas.read_csv(filepath_or_buffer=train_path,
                            names=CSV_COLUMN_NAMES,
                            header=0)

    test = pandas.read_csv(filepath_or_buffer=test_path,
                           names=CSV_COLUMN_NAMES,
                           header=0)

    train_data, train_label = train, train.pop('Species')
    test_data, test_label = test, test.pop('Species')

    feature_column_names = []

    for key in train_data.keys():
        feature_column_names.append(tensorflow.feature_column.
                                    numeric_column(key=key))

    classifier = tensorflow.estimator.DNNClassifier(
        feature_columns=feature_column_names,
        hidden_units=[10, 10],
        n_classes=3)

    classifier.train(
        input_fn=lambda: train_input_fn(train_data, train_label, 100),
        steps=1000
    )

    classifier.evaluate(
        input_fn=lambda: eval_input_fn(test_data, test_label, 100)
    )

    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_data = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    preds = classifier.predict(
        input_fn=lambda: eval_input_fn(predict_data,
                                       labels=None,
                                       batch_size=100),
    )

    for p, e in zip(preds, expected):
        class_id = p['class_ids'][0]
        print("prediction:{}, certainty: {:.4f}%, actual answer: {}".format(
            SPECIES[class_id],
            p['probabilities'][class_id] * 100,
            expected[class_id])
        )


if (__name__ == "__main__"):
    tensorflow.logging.set_verbosity(tensorflow.logging.ERROR)
    main()
