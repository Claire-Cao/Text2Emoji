import csv
from transformers import BartTokenizer
from torch.utils.data import Dataset


EmojiTypes = ['art', 'sports', 'weather',
              'animal', 'plant', 'electrical_appliances',
              'food', 'country', 'vehicle', 'experiences',
              'activities', 'feeling', 'tool', 'building',
              'family_members', 'clothes', 'career']


class Emoji2TextDataset(Dataset):
    def __init__(self, data_file_path, tokenizer_file_dir, is_training=True):
        super().__init__()

        # the tokenizer.json/ vocab.json/ merges.txt/ updated from bart-base official one
        # token_file_dir = "pretrain-models/bart-base-added-token"
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_file_dir)
        self.max_input_size = 192
        self.max_output_size = 32
        self.data_file = data_file_path
        name2id_dict = dict()
        for i in range(len(EmojiTypes)):
            name2id_dict[EmojiTypes[i]] = i

        with open(data_file_path) as csv_file:
            csvreader = csv.reader(csv_file)
            all_samples = []
            all_emoji_gt = []
            all_type_gt = []
            num_samples = 0
            for row in csvreader:
                num_samples += 1
                if num_samples == 1:
                    continue
                all_samples.append(row[0])
                all_emoji_gt.append(row[1])
                all_type_gt.append(name2id_dict[row[2]])
            total_num_samples = len(all_samples)
            print('There are {} items in the dataset.'.format(len(all_samples)))

        self.all_samples = all_samples[:int(total_num_samples * 0.9)] if is_training else \
            all_samples[int(total_num_samples * 0.9):]
        self.all_emoji_gt = all_emoji_gt[:int(total_num_samples * 0.9)] if is_training else \
            all_emoji_gt[int(total_num_samples * 0.9):]
        self.all_type_gt = all_type_gt[:int(total_num_samples * 0.9)] if is_training else \
            all_type_gt[int(total_num_samples * 0.9):]

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, index):
        sentence = self.all_samples[index]
        emoji_gt = self.all_emoji_gt[index]
        sentence_token_ids = self.tokenizer(sentence,
                                            return_tensors="pt",
                                            max_length=self.max_input_size,
                                            padding='max_length',
                                            truncation=True)  # 'attention_mask ignored'
        emoji_token_ids = self.tokenizer(emoji_gt,
                                         return_tensors="pt",
                                         max_length=self.max_output_size,
                                         padding='max_length',
                                         truncation=True)

        combined_output = {'input_ids': sentence_token_ids['input_ids'][0],
                           'input_attention_mask': sentence_token_ids['attention_mask'][0],
                           'target_ids': emoji_token_ids['input_ids'][0],
                           'target_attention_mask': emoji_token_ids['attention_mask'][0]}

        return combined_output


def data_sanity_check():
    from transformers import BartTokenizer

    # the tokenizer.json/ vocab.json/ merges.txt/ updated from bart-base official one
    token_file_dir = "pretrain-models/bart-base-added-token"

    emoji_ds = Emoji2TextDataset('./dataset/Text2Emoji/text2emoji.csv', token_file_dir)

    print(emoji_ds[0])


if __name__ == "__main__":
    data_sanity_check()




