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

    def get_label_generater(self, sample):
        return f"{sample['aspects'][0]['category']}:{sample['aspects'][0]['polarity']}"
    def get_label_classifier(self, sample):
        map = {"neutral": 1, "positive": 2, "negative": 0}
        return map[sample['aspects'][0]['polarity']]
    def get_aspect(self, sample):
        return sample['aspects'][0]['category']
    def get_polarity(self, sample):
        return sample['aspects'][0]['polarity']
    def get_input_sep(self, sample):
        return f"{sample['text']} [SEP] {self.get_aspect(sample)}"

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        count_p = 0
        count_n = 0
        for (i, ids) in enumerate(lines):
            id = "%s-%s" % (set_type, ids )
            aspect = lines[ids]['term']
            sentence = lines[ids]['sentence']
            label = lines[ids]['polarity']

            if label == "positive" or label == "+":
                count_p += 1
                label = "positive"
            elif label == "negative" or label == "-":
                count_n += 1
                label = "negative"
            else:
                label = "neutral"

            examples.append({
                "text": sentence,
                "aspects": [{
                    "category": aspect,
                    "polarity": label
                }],
            })

        return examples

