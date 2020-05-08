import numpy as np
from tqdm import tqdm


class NsmcDataPredictor:
    DATA_COLUMN = 'document'

    def __init__(self, tokenizer, data_df, seq_len):
        self.tokenizer = tokenizer
        data_df[self.DATA_COLUMN] = data_df[self.DATA_COLUMN].astype(str)
        self.data_df = data_df
        self.seq_len = seq_len

    def __convert_data(self):
        indices = []
        for i in tqdm(range(len(self.data_df))):
            ids, segments = self.tokenizer.encode(self.data_df[self.DATA_COLUMN][i], max_len=self.seq_len)
            indices.append(ids)
        indices = np.array(indices)
        return [indices, np.zeros_like(indices)]

    def load_data(self):
        return self.__convert_data()
