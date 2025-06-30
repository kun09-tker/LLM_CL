
import torch
import numpy as np

from sklearn.metrics import accuracy_score, f1_score

def classifier_metrics(preds, labels, average="macro", device='cpu'):
    preds = preds.to(device)
    labels = labels.to(device)
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average=average)  # hoặc "macro", tùy bài toán
    return acc, f1

def generater_metrics(y_true, y_pred, average="macro"):
      total_tp = 0
      total_pred = 0
      total_gt = 0

      macro_prec_list = []
      macro_rec_list = []
      macro_f1_list = []

      for gt, pred in zip(y_true, y_pred):
          gt_list = gt.split('; ') if gt.strip() != "" else []
          pred_list = pred.split('; ') if pred.strip() != "" else []

          sample_tp = 0

          total_pred += len(pred_list)
          total_gt += len(gt_list)

          for gt_val in gt_list:
              for pred_val in pred_list:
                  if pred_val in gt_val or gt_val in pred_val:
                      sample_tp += 1
                      break
          # else:
          #     for gt_val in gt_list:
          #         parts = gt_val.split(':')
          #         if len(parts) < 3:
          #             continue
          #         gt_asp, gt_op, gt_sent = parts[0], parts[1], parts[2]
          #         for pred_val in pred_list:
          #             pr_parts = pred_val.split(':')
          #             if len(pr_parts) < 3:
          #                 continue
          #             pr_asp, pr_op, pr_sent = pr_parts[0], pr_parts[1], pr_parts[2]
          #             if pr_asp in gt_asp and pr_op in gt_op and gt_sent == pr_sent:
          #                 sample_tp += 1
          #                 break

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

      if average == "macro":
          return acc, f1_macro
      elif average == "micro":
          return acc, f1_micro
      else:
          return acc, f1_macro