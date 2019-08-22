# functions for handling wordbanks

import re
from avltree import TokenAVL

class Token():
    dict_re = r'(?P<page>\d+)\.\(?P<row>[\d+\])(?P<col>[=-])'
    table_re = r'(?P<row>\[\d+\])\^'

    def __init__(self, s, plaintext='<unk>'):
        self.raw = s.strip()
        self.plaintext = plaintext
        self.ciphertype = None # which type of cipher this uses. one of: None (unparsed), dict or table

        # parse
        m = re.match(dict_re, self.raw):
        if m:
            self.ciphertype = 'dict'
            self.page = int(m.group('page'))
            self.row = int(m.group('row'))
            self.col = {'-': 0, '=': 1}[m.group('col')] # - is col 1, = is col 2
        m = re.match(table_re, self.raw)
        if m:
            self.ciphertype = 'table'
            self.row = m.group('row')

    def __sub__(self, other):
        '''get the number of words between this and another token'''

        if self.ciphertype != 'dict' or other.ciphertype != 'dict': # can't compare
            return None
        return int(self) - int(other)

    def __int__(self):
        return (2 * 29 * (self.page - 1) # TODO: change 29 when using other wordbanks
                + 29 * self.col
                + (self.row - 1))

    def __hash__(self):
        return hash(self.raw) # need to be able to compare tokens with plaintext set and those without. this makes that equivalence true

    def __str__(self):
        return self.plaintext

class Wordbank():
    def __init__(self):
        self._dict = {} # mapping Token -> word
        self._tree = TokenAVL() # avl tree storing tokens. used too keep track of ranges

    def load(self, filename):
        with open(filename) as f:
            next(f) # skip header line

            for line in f:
                try:
                    location, source, word = line.strip().split('\t')
                except ValueError:
                    pass # blank line; skip

                if source_filter is None or source in source_filter:
                    t = Token(location, word)
                    self._dict[t] = word
                    self._tree.insert(t)

    # imaginary words that would go at the beginning and end of a dictionary
    left_dummy = Token('1.[1]-', '')
    right_dummy = Token('780.[1]-', 'zzzzzzzz')     # the last attested word is 44,312 = yourself. it will only affect words after "yourself" in the dictionary, so doesn't really matter

    def query_range(self, query_token):
        '''return the approximate position in the range'''
        if query_token.ciphertype is None or query_token.ciphertype == 'table':
            return None

        left, right = self._tree.find_neighbours(query_token)

        # error handling for beginning or end of string
        if left is None and right is None:
            raise ValueError('no tokens have been inserted')
        elif left is None:
            left = left_dummy
        elif right is None:
            right = right_dummy

        return left, right