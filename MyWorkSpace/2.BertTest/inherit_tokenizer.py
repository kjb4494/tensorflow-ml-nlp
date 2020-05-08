from keras_bert import Tokenizer


# Tokenizer 상속
class InheritTokenizer(Tokenizer):
    # 원문
    # def _tokenize(self, text):
    #     if not self._cased:
    #         text = unicodedata.normalize('NFD', text)
    #         text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
    #         text = text.lower()
    #     spaced = ''
    #     for ch in text:
    #         if self._is_punctuation(ch) or self._is_cjk_character(ch):
    #             spaced += ' ' + ch + ' '
    #         elif self._is_space(ch):
    #             spaced += ' '
    #         elif ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
    #             continue
    #         else:
    #             spaced += ch
    #     tokens = []
    #     for word in spaced.strip().split():
    #         tokens += self._word_piece_tokenize(word)
    #     return tokens

    # _tokenize 오버라이딩
    def _tokenize(self, text):
        if not self._cased:
            text = text.lower()
        spaced = ''
        for ch in text:
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                spaced += ' ' + ch + ' '
            elif self._is_space(ch):
                spaced += ' '
            #
            elif ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch
        tokens = []
        for word in spaced.strip().split():
            tokens += self._word_piece_tokenize(word)
        return tokens
