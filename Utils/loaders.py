import os
import json
import pandas as pd
# from datasets import Dataset
# from datasets.dataset_dict import DatasetDict

class SimpleDatasetLoader:
    def __init__(self, train_df: pd.DataFrame = None,
                 test_df: pd.DataFrame = None,
                 val_df: pd.DataFrame = None,
                 sample_size: float = 1.0):
        if train_df is not None and sample_size < 1.0 and sample_size > 0:
            print(f"Sampling {sample_size * 100:.2f}% of the training data.")
            self.train_df = train_df.sample(frac=sample_size, random_state=1999)
        else:
            self.train_df = train_df

        self.test_df = test_df
        self.val_df = val_df

        if self.train_df is not None:
             print(f"Initialized with {len(self.train_df)} training records.")
        if self.test_df is not None:
             print(f"Initialized with {len(self.test_df)} test records.")
        if self.val_df is not None:
             print(f"Initialized with {len(self.val_df)} validation records.")

    @staticmethod
    def load_from_json(train_path: str = None,
                       test_path: str = None,
                       val_path: str = None,
                       domain_name: str = "",
                       sample_size: float = 1.0) -> 'SimpleDatasetLoader':
        train_df, test_df, val_df = None, None, None

        def _read_json_to_df(path, file_type):
            if path and os.path.exists(path):
                print(f"Loading {file_type} data from: {path}")
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data_all_domains = json.load(f)
                        data = data_all_domains.get(domain_name, [])

                    if not isinstance(data, list):
                        raise ValueError(f"JSON content in {path} is not a list as expected.")

                    df = pd.DataFrame(data)
                    print(f"Successfully loaded {len(df)} records from {file_type} file.")
                    return df

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from {path}. Check file format and encoding. Details: {e}")
                    raise
                except ValueError as e:
                    print(f"Error processing JSON data from {path}: {e}")
                    raise
                except Exception as e:
                    print(f"An unexpected error occurred while reading {path}: {e}")
                    raise
            elif path:
                raise FileNotFoundError(f"{file_type.capitalize()} file not found at: {path}")
            return None

        try:
            train_df = _read_json_to_df(train_path, "training")
            test_df = _read_json_to_df(test_path, "test")
            val_df = _read_json_to_df(val_path, "validation")
        except (FileNotFoundError, json.JSONDecodeError, ValueError, Exception) as e:
             print(f"Halting execution due to data loading error: {e}")
             return None

        return SimpleDatasetLoader(train_df=train_df, test_df=test_df, val_df=val_df, sample_size=sample_size)

    def create_data_in_category_sentiment_format(self,
                                                df: pd.DataFrame,
                                                text_col: str = 'text',
                                                aspect_col: str = 'aspects',
                                                bos_instruction: str = '',
                                                eos_instruction: str = '',
                                                number_of_sample: int = -1) -> pd.DataFrame:
        if df is None:
             raise ValueError("Input DataFrame cannot be None for formatting")

        if df.empty:
            print(f"Warning: Input DataFrame for formatting is empty. Returning it as is.")
            if 'text' not in df.columns: df['text'] = None
            if 'labels' not in df.columns: df['labels'] = None
            return df

        valid_aspect_rows = df[df[aspect_col].apply(lambda x: isinstance(x, list) and len(x) > 0)]

        if not valid_aspect_rows.empty:
            first_valid_aspect_row = valid_aspect_rows.iloc[0]
            first_aspect_list = first_valid_aspect_row[aspect_col]

            if not isinstance(first_aspect_list, list):
                 raise TypeError(f"Column '{aspect_col}' should contain lists, but found {type(first_aspect_list)} in the first valid row.")
            if first_aspect_list and isinstance(first_aspect_list[0], dict):
                 first_aspect_dict = first_aspect_list[0]
                 if 'category' not in first_aspect_dict or 'polarity' not in first_aspect_dict:
                     raise KeyError(f"Dictionaries in '{aspect_col}' must contain 'category' and 'polarity' keys. Found keys: {list(first_aspect_dict.keys())}")
            elif first_aspect_list and not isinstance(first_aspect_list[0], dict):
                 raise TypeError(f"Elements within the list in '{aspect_col}' should be dictionaries, but found {type(first_aspect_list[0])}.")

        def format_labels(aspect_list):
            if not isinstance(aspect_list, list):
                return ""

            pairs = []
            for aspect in aspect_list:
                if isinstance(aspect, dict):
                    category = aspect.get('category')
                    polarity = aspect.get('polarity')
                    if category and polarity:
                         pairs.append(f"{category}:{polarity}")
            return ', '.join(pairs)

        df['labels'] = df[aspect_col].apply(format_labels)

        df['text'] = df[text_col].apply(lambda x: bos_instruction + str(x) + eos_instruction if pd.notna(x) else bos_instruction + eos_instruction)

        if number_of_sample != -1:
          return df[:number_of_sample]
        else:
          return df

    # def set_data_for_training(self, tokenize_function) -> tuple:
    #     dataset_dict = {}
    #     prepared = True

    #     for split_name, df in [('train', self.train_df), ('test', self.test_df), ('validation', self.val_df)]:
    #         if df is not None:
    #              if 'labels' not in df.columns or 'text' not in df.columns:
    #                  print(f"Warning: {split_name.capitalize()} data is not formatted yet (missing 'text' or 'labels' column).")
    #                  prepared = False
    #              else:
    #                  dataset_dict[split_name] = Dataset.from_pandas(df)

    #     raw_dataset_dict = None
    #     tokenized_dataset_dict = None

    #     if dataset_dict and prepared:
    #         raw_dataset_dict = DatasetDict(dataset_dict)
    #         print("Applying tokenization...")
    #         try:
    #             example_split_name = next(iter(raw_dataset_dict))
    #             cols_to_keep = ['input_ids', 'attention_mask', 'labels']
    #             cols_to_remove = [col for col in raw_dataset_dict[example_split_name].column_names if col not in cols_to_keep]

    #             tokenized_dataset_dict = raw_dataset_dict.map(
    #                 tokenize_function,
    #                 batched=True,
    #                 remove_columns=cols_to_remove
    #             )
    #             print("Tokenization complete.")
    #         except Exception as e:
    #              print(f"Error during tokenization: {e}")
    #              print("Please check your tokenize_function and the data structure.")
    #              tokenized_dataset_dict = None

    #     elif not prepared and dataset_dict:
    #          print("Skipping tokenization because data was not formatted correctly.")
    #          raw_dataset_dict = DatasetDict(dataset_dict)
    #          tokenized_dataset_dict = DatasetDict()
    #     else:
    #         print("Warning: No valid data splits available or prepared for tokenization.")
    #         raw_dataset_dict = DatasetDict()
    #         tokenized_dataset_dict = DatasetDict()

    #     return raw_dataset_dict, tokenized_dataset_dict

class InstructionsHandler:
    def __init__(self):
        self.aspe = {}

    def load_instruction_set1(self):
        self.aspe['bos_instruct1'] = "Given a Sentence, you should extract all aspect terms and give a corresponding polarity. The format is \"terms1: polarity1; terms2: polarity2\". Sentence: "
        self.aspe['eos_instruct'] = ''