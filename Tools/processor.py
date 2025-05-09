import os
import json

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        """Reads a json file for tasks in sentiment analysis."""
        with open(input_file) as f:
            return json.load(f)

class AscProcessor(DataProcessor):
    """Processor for the SemEval Aspect Sentiment Classification."""

    def get_train_examples(self, data_dir, fn="train.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "train")

    def get_dev_examples(self, data_dir, fn="dev.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "dev")

    def get_test_examples(self, data_dir, fn="test.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "test")

    def get_labels(self):
        """See base class."""
        return ["positive", "negative", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, ids) in enumerate(lines):
            id = "%s-%s" % (set_type, ids )
            aspect = lines[ids]['term']
            sentence = lines[ids]['sentence']
            label = lines[ids]['polarity']

            if label == "+":
                label = "positive"
            elif label == "-":
                label = "negative"
            else:
                label = "neutral"

            if label == "positive":
                label = 1
            elif label == "negative":
                label = -1
            else:
                label = 0

            examples.append((f'"{sentence}":{aspect}', label))
        return examples

# class InputExample(object):
#     """A single training/test example for simple sequence classification."""

#     def __init__(self, id=None, sentence=None, aspect=None, label=None):
#         self.id = id
#         self.sentence = sentence
#         self.aspect = aspect
#         self.label = label