from general_beamsearch import LanguageModel, LMScore # for writing a general LM
from passes import split_ciphertext

class OracleLanguageModel(LanguageModel):
    def __init__(self, gold_file):

        with open(gold_file) as fh:
            content = fh.read()
            self.tokens = split_ciphertext(content)

            print('oracle tokens', self.tokens)

    def encode(self, word):
        '''Tokenize and encode a word. Should return a list'''
        return [word]

    def decode(self, tokens):
        '''Decode a series of tokens. Return a string'''
        return ' '.join(tokens)

    def score(self, context, words):
        '''Take a list of tokens as context and list of words. Return a list of LMScore for each word'''


        # multiply score by an arbitrarily large number to ignore the lattice model
        # this should score the entire sentence, but it doesn't, so need to set beam width 1.
        return [LMScore(tokens=[word], score=100 * (word == (self.tokens + ['a']*3)[len(context)])) for word in words]

