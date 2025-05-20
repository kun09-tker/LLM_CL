import read_file
import utils

def render(file_path):
    # Đọc và xử lý file
    data = read_file.read_training_log(file_path)
    if data:
        # Tính trung bình các metrics theo epoch
        averages = utils.compute_epoch_averages(data)
        print("Dữ liệu trung bình theo epoch:")
        for domain in averages:
            print(f"\nDomain: {domain}")
            for metric in averages[domain]:
                print(f"{metric}: {averages[domain][metric]}")
    else:
        print ("Fail to extract file")