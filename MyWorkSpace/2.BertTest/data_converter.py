import numpy as np
from tqdm import tqdm


class NsmcDataConverter:
    DATA_COLUMN = 'document'
    LABEL_COLUMN = 'label'

    def __init__(self, tokenizer, pandas_data_frame, seq_len):
        self.tokenizer = tokenizer
        self.data_df = pandas_data_frame
        self.seq_len = seq_len

    def __convert_data(self):
        indices, targets = [], []
        for i in tqdm(range(len(self.data_df))):
            ids, segments = self.tokenizer.encode(self.data_df[self.DATA_COLUMN][i], max_len=self.seq_len)
            indices.append(ids)
            targets.append(self.data_df[self.LABEL_COLUMN][i])
        items = list(zip(indices, targets))
        indices, targets = zip(*items)
        indices = np.array(indices)
        return [indices, np.zeros_like(indices)], np.array(targets)

    def load_data(self):
        self.data_df[self.DATA_COLUMN] = self.data_df[self.DATA_COLUMN].astype(str)
        data_x, data_y = self.__convert_data()
        return data_x, data_y


class SentenceConverter:
    def __init__(self, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __convert_data(self, data):
        indices = []
        for i in tqdm(range(len(data))):
            ids, segments = self.tokenizer.encode(data[i], max_len=self.seq_len)
            indices.append(ids)
        indices = np.array(indices)
        return [indices, np.zeros_like(indices)]

    def load_data(self, sentences):
        return self.__convert_data(sentences)
