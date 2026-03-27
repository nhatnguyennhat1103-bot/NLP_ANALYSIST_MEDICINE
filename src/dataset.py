import torch
from torch.utils.data import Dataset

class MedicalNERDataset(Dataset):
    def __init__(self, file_path, tokenizer, label2id, max_len=128):
        self.data = self.load_data(file_path)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def load_data(self, file_path):
        # Đọc file txt định dạng Word - Label của bạn
        sentences, labels = [], []
        curr_sent, curr_labels = [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() == "":
                    if curr_sent:
                        sentences.append(curr_sent)
                        labels.append(curr_labels)
                        curr_sent, curr_labels = [], []
                else:
                    word, label = line.strip().split()
                    curr_sent.append(word)
                    curr_labels.append(label)
        return sentences, labels

    def __getitem__(self, item):
        words = self.data[0][item]
        labels = self.data[1][item]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        # Căn chỉnh nhãn cho các subwords
        word_ids = encoding.word_ids() 
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100) # Bỏ qua các token đặc biệt như [CLS], [SEP]
            else:
                label_ids.append(self.label2id[labels[word_idx]])

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_ids)
        }

    def __len__(self):
        return len(self.data[0])