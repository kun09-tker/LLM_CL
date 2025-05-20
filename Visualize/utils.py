import numpy as np

def compute_epoch_averages(data):
    # Dictionary để lưu trữ trung bình theo epoch: {domain: {metric: [avg_values]}}
    averages = {'rest': {'Loss_train': [], 'Acc_domain_variant': [], 'F1_macro_domain_variant': [],
                         'Acc_domain_invariant': [], 'F1_macro_domain_invariant': []},
                'laptop': {'Loss_train': [], 'Acc_domain_variant': [], 'F1_macro_domain_variant': [],
                           'Acc_domain_invariant': [], 'F1_macro_domain_invariant': []}}

    for domain in data:
        for epoch in range(1, 31):
            for metric in averages[domain]:
                batch_values = data[domain][epoch][metric]
                if batch_values:  # Chỉ tính trung bình nếu có dữ liệu
                    avg_value = np.mean(batch_values)
                else:
                    avg_value = np.nan  # Gán NaN nếu không có batch nào
                averages[domain][metric].append(avg_value)

    return averages