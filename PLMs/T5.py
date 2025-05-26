import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import (
    DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)

class T5Generator:
    def __init__(self, model_checkpoint, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        self.device = device
        self.model.to(device)

    # def get_device(self):
    #     if torch.cuda.is_available():
    #         return 'cuda'
    #     elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    #         return 'mps'
    #     else:
    #         return 'cpu'

    def tokenize_function_inputs(self, sample):
        model_inputs = self.tokenizer(sample['text'], max_length=512, truncation=True)
        labels = self.tokenizer(sample["labels"], max_length=64, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def train(self, tokenized_datasets, **kwargs):
        args = Seq2SeqTrainingArguments(
            report_to=[],
            **kwargs
        )

        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets.get("validation"),
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        print("Trainer device:", trainer.args.device)

        torch.cuda.empty_cache()
        print('\nModel training started ....')
        trainer.train()

        trainer.save_model()
        return trainer

    def get_labels(self, tokenized_dataset, batch_size=4, max_length=128, sample_set='train'):
        def collate_fn(batch):
            input_ids = [torch.tensor(example['input_ids']) for example in batch]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            return input_ids

        dataloader = DataLoader(tokenized_dataset[sample_set], batch_size=batch_size, collate_fn=collate_fn)
        predicted_output = []
        self.model.to(self.device)
        print('Model loaded to:', self.device)

        for batch in tqdm(dataloader):
            batch = batch.to(self.device)
            output_ids = self.model.generate(batch, max_length=max_length)
            output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            predicted_output.extend(output_texts)
        return predicted_output

    def get_metrics(self, y_true, y_pred, is_triplet_extraction=False):
        total_tp = 0
        total_pred = 0
        total_gt = 0

        macro_prec_list = []
        macro_rec_list = []
        macro_f1_list = []

        for gt, pred in zip(y_true, y_pred):
            gt_list = gt.split(', ') if gt.strip() != "" else []
            pred_list = pred.split(', ') if pred.strip() != "" else []

            sample_tp = 0

            total_pred += len(pred_list)
            total_gt += len(gt_list)

            if not is_triplet_extraction:
                for gt_val in gt_list:
                    for pred_val in pred_list:
                        if pred_val in gt_val or gt_val in pred_val:
                            sample_tp += 1
                            break
            else:
                for gt_val in gt_list:
                    parts = gt_val.split(':')
                    if len(parts) < 3:
                        continue
                    gt_asp, gt_op, gt_sent = parts[0], parts[1], parts[2]
                    for pred_val in pred_list:
                        pr_parts = pred_val.split(':')
                        if len(pr_parts) < 3:
                            continue
                        pr_asp, pr_op, pr_sent = pr_parts[0], pr_parts[1], pr_parts[2]
                        if pr_asp in gt_asp and pr_op in gt_op and gt_sent == pr_sent:
                            sample_tp += 1
                            break

            total_tp += sample_tp

            if len(pred_list) > 0:
                sample_prec = sample_tp / len(pred_list)
            else:
                sample_prec = 1.0 if len(gt_list) == 0 else 0.0

            if len(gt_list) > 0:
                sample_rec = sample_tp / len(gt_list)
            else:
                sample_rec = 1.0 if len(pred_list) == 0 else 0.0

            if sample_prec + sample_rec > 0:
                sample_f1 = 2 * sample_prec * sample_rec / (sample_prec + sample_rec)
            else:
                sample_f1 = 0.0

            macro_prec_list.append(sample_prec)
            macro_rec_list.append(sample_rec)
            macro_f1_list.append(sample_f1)

        precision_macro = np.mean(macro_prec_list) if macro_prec_list else 0
        recall_macro = np.mean(macro_rec_list) if macro_rec_list else 0
        f1_macro = np.mean(macro_f1_list) if macro_f1_list else 0

        precision_micro = total_tp / total_pred if total_pred > 0 else 0
        recall_micro = total_tp / total_gt if total_gt > 0 else 0
        f1_micro = (2 * precision_micro * recall_micro / (precision_micro + recall_micro)
                    if (precision_micro + recall_micro) > 0 else 0)

        correct_matches = 0
        for gt, pred in zip(y_true, y_pred):
            if gt.strip() == pred.strip():
                correct_matches += 1
        acc = correct_matches / len(y_true) if len(y_true) > 0 else 0

        report = None

        return (precision_macro, recall_macro, f1_macro,
                precision_micro, recall_micro, f1_micro, acc, report)