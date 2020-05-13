import codecs
from keras_bert import Tokenizer


def get_token_dict(vocab_path):
    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token = '##' + token.replace('_', '') if '_' in token else token
            token_dict[token] = len(token_dict)
    return token_dict


def get_reverse_token_dict(token_dict):
    return {v: k for k, v in token_dict.items()}


# Tokenizer 상속
class InheritTokenizer(Tokenizer):
    def _tokenize(self, text):
        if not self._cased:
            text = text.lower()
        spaced = ''
        for ch in text:
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                spaced += ' ' + ch + ' '
            elif self._is_space(ch):
                spaced += ' '
            elif ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch
        tokens = []
        for word in spaced.strip().split():
            tokens += self._word_piece_tokenize(word)
        return tokens
