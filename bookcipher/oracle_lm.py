from general_beamsearch import LanguageModel, LMScore # for writing a general LM
from passes import split_ciphertext

class OracleLanguageModel(LanguageModel):
    def __init__(self, gold_file):

        with open(gold_file) as fh:
            content = fh.read()
            self.tokens = split_ciphertext(content)

    def encode(self, word):
        '''Tokenize and encode a word. Should return a list'''
        return [word]

    def decode(self, tokens):
        '''Decode a series of tokens. Return a string'''
        return ' '.join(tokens)

    def score(self, index, words):
        '''
        Return a list of LMScore for the `index`th word, with score indicating if the token is correct
        '''
        result = [LMScore(tokens=[word + ('*' if word != self.tokens[index] else '')], score=(word == self.tokens[index])) for word in words]
        return result
