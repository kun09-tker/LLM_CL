import json
import re
from collections import defaultdict

def parse_file_to_json(file_path, output_path):
    # Initialize the main structure
    result = {"show": []}
    current_domain = None
    current_epoch = None
    # step1_training = defaultdict(lambda: {"Data": []})
    # step1_validating = defaultdict(list)
    # step2_data = defaultdict(lambda: {"Data": []})
    # step3_data = {}

    # Regular expressions for parsing lines
    step1_batch_re = re.compile(r"Step1 - Epoch (\d+)/\d+ - batch (\d+) - domain_name (\w+),")
    step1_epoch_re = re.compile(r"Step1 - Epoch (\d+)/\d+ - domain_name (\w+),")
    step2_batch_re = re.compile(r"Step2 - Epoch (\d+)/\d+ - batch (\d+) - domain (\w+),")
    step2_epoch_re = re.compile(r"Step2 - Epoch (\d+)/\d+ - domain (\w+),")
    step3_re = re.compile(r"Step 3 Loss test: ([\d.]+) - Acc test: ([\d.]+) - F1_Macro: ([\d.]+)")
    metric_re = re.compile(r"([A-Za-z\s_1]+):\s+([\d]+\.\d+(?:e[-+]\d+)?)")

    # Track domains to avoid duplicates
    domain_data = defaultdict(lambda: {
        "Step 1": {"Training": [], "Validating": []},
        "Step 2": [],
        "Step 3": {}
    })
    epoch_data_step1 = {}
    epoch_data_step2 = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            # line = lines[i].strip()
            # print(line)
            # print(i)
            if not lines[i].strip() or lines[i].strip().startswith("==") or lines[i].strip() in ["**Start***", ""]:
                i += 1
            # Match Step 1 batch
            match = step1_batch_re.match(lines[i].strip())
            if match:
                # print(match)
                epoch, batch, domain = int(match.group(1)), int(match.group(2)), match.group(3)
                if epoch != current_epoch:
                    epoch_data_step1["Epoch"] = epoch
                    epoch_data_step1["Data"] = []
                    current_epoch = epoch
                    # if current_domain:
                    #     break

                if "Data" in epoch_data_step1.keys():
                    # print(epoch, batch, domain)
                    batch_data = {}
                    batch_data["Batch"] = batch
                    for _ in range(5):
                        i += 1
                        metric_match = metric_re.match(lines[i].strip())
                        # print(i)
                        if metric_match:
                            key, value = metric_match.groups()
                            batch_data[key.replace(" ", "_").strip()] = float(value)
                    epoch_data_step1["Data"].append(batch_data)
                    # print(epoch_data)

            match = step1_epoch_re.match(lines[i].strip())
            if match:
                epoch, domain = int(match.group(1)), match.group(2)
                validate = {}
                validate["Epoch"] = epoch
                for _ in range(5):
                    i += 1
                    metric_match = metric_re.match(lines[i].strip())
                    # print(i)
                    if metric_match:
                        key, value = metric_match.groups()
                        if "val" in key:
                            validate[key.replace(" ", "_").strip()] = float(value)
                        else:
                            epoch_data_step1[key.replace(" ", "_").strip()] = float(value)

                if "Loss_val" in validate.keys():
                    domain_data[domain]["Step 1"]["Validating"].append(validate)
                else:
                    domain_data[domain]["Step 1"]["Training"].append(epoch_data_step1)
                    epoch_data_step1 = {}

            match = step2_batch_re.match(lines[i].strip())
            if match:
                # print(match)
                epoch, batch, domain = int(match.group(1)), int(match.group(2)), match.group(3)
                if epoch != current_epoch:
                    epoch_data_step2["Epoch"] = epoch
                    epoch_data_step2["Data"] = []
                    current_epoch = epoch
                    # if current_domain:
                    #     break

                if "Data" in epoch_data_step2.keys():
                    # print(epoch, batch, domain)
                    batch_data = {}
                    batch_data["Batch"] = batch
                    for _ in range(3):
                        i += 1
                        metric_match = metric_re.match(lines[i].strip())
                        # print(i)
                        if metric_match:
                            key, value = metric_match.groups()
                            batch_data[key.replace(" ", "_").strip()] = float(value)
                    epoch_data_step2["Data"].append(batch_data)
                    # print(epoch_data)

            match = step2_epoch_re.match(lines[i].strip())
            if match:
                epoch, domain = int(match.group(1)), match.group(2)
                for _ in range(3):
                    i += 1
                    metric_match = metric_re.match(lines[i].strip())
                    # print(i)
                    if metric_match:
                        key, value = metric_match.groups()
                        epoch_data_step2[key.replace(" ", "_").strip()] = float(value)

                domain_data[domain]["Step 2"].append(epoch_data_step2)
                epoch_data_step2 = {}
                # # print(domain_data)
            match = step3_re.match(lines[i].strip())
            if match:
                Loss, Acc, F1_score = float(match.group(1)), float(match.group(2)), float(match.group(3))
                domain_data[domain]["Step 3"]["Loss_test"] = Loss
                domain_data[domain]["Step 3"]["Acc_test"] = Acc
                domain_data[domain]["Step 3"]["F1_macro_test"] = F1_score

            i += 1

        domain_data_dict = dict(domain_data)
        # Write to a JSON file
        with open(output_path, 'w') as file:
            json.dump(domain_data_dict, file, indent=4)