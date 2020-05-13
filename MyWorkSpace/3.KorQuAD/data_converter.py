import numpy as np
import keras
from tqdm import tqdm


class KorQuADDataConverter:
    DATA_COLUMN = "context"
    QUESTION_COLUMN = "question"
    TEXT = "text"

    def __init__(self, tokenizer, pandas_data_frame, seq_len):
        self.tokenizer = tokenizer
        self.data_df = pandas_data_frame
        self.data_df[self.DATA_COLUMN] = self.data_df[self.DATA_COLUMN].astype(str)
        self.data_df[self.QUESTION_COLUMN] = self.data_df[self.QUESTION_COLUMN].astype(str)
        self.seq_len = seq_len

    def __convert_data(self):
        indices, segments, target_start, target_end = [], [], [], []
        for i in tqdm(range(len(self.data_df))):
            ids, segment = self.tokenizer.encode(
                self.data_df[self.QUESTION_COLUMN][i],
                self.data_df[self.DATA_COLUMN][i],
                max_len=self.seq_len
            )
            text = self.tokenizer.encode(self.data_df[self.TEXT][i])[0]
            text_slide_len = len(text[1:-1])
            ans_start = self.seq_len
            ans_end = self.seq_len
            for j in range(1, len(ids) - text_slide_len - 1):
                if text[1:-1] == ids[j:j + text_slide_len]:
                    ans_start = 1
                    ans_end = j + text_slide_len - 1
                    break
            indices.append(ids)
            segments.append(segment)
            target_start.append(ans_start)
            target_end.append(ans_end)
        indices_x = np.array(indices)
        segments = np.array(segments)
        target_start = np.array(target_start)
        target_end = np.array(target_end)

        del_list = np.where(target_start != self.seq_len)[0]
        indices_x = indices_x[del_list]
        segments = segments[del_list]
        target_start = target_start[del_list]
        target_end = target_end[del_list]

        train_y_0 = keras.utils.to_categorical(target_start, num_classes=self.seq_len, dtype='int64')
        train_y_1 = keras.utils.to_categorical(target_end, num_classes=self.seq_len, dtype='int64')
        train_y_cat = [train_y_0, train_y_1]

        return [indices_x, segments], train_y_cat

    def load_data(self):
        return self.__convert_data()


class KorQuADPredDataConverter:
    def __init__(self, tokenizer, question, doc, seq_len):
        self.tokenizer = tokenizer
        self.question = question
        self.doc = doc
        self.seq_len = seq_len

    def __convert_data(self):
        indices, segments = [], []
        ids, segment = self.tokenizer.encode(self.question, self.doc, max_len=self.seq_len)
        indices.append(ids)
        segments.append(segment)
        indices_x = np.array(indices)
        segments = np.array(segments)
        return [indices_x, segments]

    def load_data(self):
        return self.__convert_data()
