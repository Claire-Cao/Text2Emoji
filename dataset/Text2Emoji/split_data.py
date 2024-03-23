import numpy as np
import csv


def generate_data(csv_file_path):
    """
    In the Text2Emoji dataset, there are 503687 samples. We split this dataset into
    train/val/test, in the ratio of 8/1/1. More precisely, we randomly select 402950
    samples for training, 50000 for val and 50737 for test.
    """

    shuffled_index = np.random.permutation(np.arange(503687))
    train_index = shuffled_index[:402950]
    val_index = shuffled_index[402950: 452950]
    test_index = shuffled_index[452950:]

    with open(csv_file_path) as csv_file:
        csvreader = csv.reader(csv_file)
        all_samples = []
        num_samples = 0
        for row in csvreader:
            num_samples += 1
            if num_samples == 1:
                continue
            all_samples.append(row)

    # field names
    fields = ['text', 'emoji', 'topic']

    train_data = []
    for i in train_index:
        train_data.append({fields[0]: all_samples[i][0],
                           fields[1]: all_samples[i][1],
                           fields[2]: all_samples[i][2]})

    val_data = []
    for i in val_index:
        val_data.append({fields[0]: all_samples[i][0],
                         fields[1]: all_samples[i][1],
                         fields[2]: all_samples[i][2]})

    test_data = []
    for i in test_index:
        test_data.append({fields[0]: all_samples[i][0],
                          fields[1]: all_samples[i][1],
                          fields[2]: all_samples[i][2]})

    # name of csv file
    train_filename = "text2emoji_train.csv"
    val_filename = "text2emoji_val.csv"
    test_filename = "text2emoji_test.csv"

    # writing to csv file
    with open(train_filename, 'w') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # writing headers (field names)
        writer.writeheader()

        # writing data rows
        writer.writerows(train_data)

    with open(val_filename, 'w') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # writing headers (field names)
        writer.writeheader()

        # writing data rows
        writer.writerows(val_data)

    with open(test_filename, 'w') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # writing headers (field names)
        writer.writeheader()

        # writing data rows
        writer.writerows(test_data)


if __name__ == "__main__":
    generate_data('text2emoji.csv')

