# functions for handling wordbanks

import re
from functools import total_ordering

from avltree import TokenAVL

@total_ordering
class Token():
    dict_re = r'(?P<page>\d+)\.\[(?P<row>\d+)\](?P<col>[=-])'
    table_re = r'\[(?P<row>\d+)\]\^'

    def __init__(self, raw_string, plaintext='<unk>'):
        self.raw = raw_string.strip(' ')
        self.plaintext = plaintext
        self.ciphertype = None # which type of cipher this uses. one of: None (unparsed), dict or table

        self.source = ''
        self.prob = 0

        # parse
        m_dict = re.match(Token.dict_re, self.raw)
        m_table = re.match(Token.table_re, self.raw)
        if m_dict:
            self.ciphertype = 'dict'
            self.page = int(m_dict.group('page'))
            self.row = int(m_dict.group('row'))
            self.col = {'-': 0, '=': 1}[m_dict.group('col')] # - is col 1, = is col 2

            if self.page > 780: # assume anything outside of this range is a mistake
                self.ciphertype = None
                del self.page
                del self.row
                del self.col
                self.plaintext = '<unk>'

        elif m_table:
            self.ciphertype = 'table'
            self.row = int(m_table.group('row'))

            if self.row < 160 or self.row >= 1240: # outside of this range, the table isn't alphabetical. it's proper nouns and other stuff
                self.ciphertype = None
                del self.row
                self.plaintext = '<unk>'

        else:
            self.plaintext = self.raw
            self.source = 'literal'
            self.prob = 1

    def is_unk(self):
        return self.plaintext == '<unk>'

    def __sub__(self, other):
        '''get the number of words between this and another token'''

        if self.ciphertype != 'dict' or other.ciphertype != 'dict': # can't compare
            return None
        return int(self) - int(other)

    def __gt__(self, other):
        return int(self) > int(other)

    def __eq__(self, other):
        if self.ciphertype == 'dict' or self.ciphertype == 'table':
            return int(self) == int(other)
        else:
            return self.raw == other.raw

    def __int__(self):
        if self.ciphertype == 'dict':
            return (2 * 29 * (self.page - 1) # TODO: change 29 when using other wordbanks
                    + 29 * self.col
                    + (self.row - 1))
        elif self.ciphertype == 'table':
            return self.row
        else:
            return hash(self.raw) # need a reproducible, unique way to identify unprocessed tokens

    def __hash__(self):
        return hash((self.ciphertype, int(self))) # need to be able to compare tokens with plaintext set and those without. this makes that equivalence true

    def __str__(self):
        return self.plaintext

    def __repr__(self):
        return "<Token(raw_string='{}', plaintext='{}')>".format(self.raw, self.plaintext)

    def to_html(self):
        if self.plaintext == '\n':
            return '''<p/>''' # easy way to make a line break
        else:
            if self.source == 'literal':
                style = 'background-color:turquoise;'
            elif self.source == 'wordbank':
                style = 'background-color:lightblue;'
            else:
                style = 'background-color: rgb(0,{},0);'.format(127 + self.prob*128)

            return '''
                <div class="tooltip" style="{style}">{plaintext}
                    <span class="tooltiptext">
                        raw:{raw}<br>p:{prob}<br>source:{source}
                    </span>
                </div>
            '''.format(plaintext=self.plaintext, source=self.source, raw=self.raw, prob=self.prob, style=style)

class Wordbank():
    def __init__(self):
        self._dict = {} # mapping Token -> word
        self._dict_tree = TokenAVL() # avl tree storing tokens for those encypted with the dictionary method. used to keep track of ranges
        self._table_tree = TokenAVL() # same as above for the table method

    def load(self, filename):
        with open(filename) as f:
            next(f) # skip header line

            for line in f:
                if line.startswith('#'):
                    continue
                try:
                    location, _source, word = line.strip().split('\t')
                except ValueError:
                    continue # blank line; skip

                # if source_filter is None or source in source_filter:
                t = Token(location, word)
                self._dict[t] = word

                if t.ciphertype == 'table':
                    self._table_tree.insert(t)
                if t.ciphertype == 'dict':
                    self._dict_tree.insert(t)

    # imaginary words that would go at the beginning and end of a dictionary
    left_dict_dummy = Token('1.[1]-', '-')
    right_dict_dummy = Token('780.[1]-', 'zzzzzzz')     # the last attested word is 44,312 = yourself. it will only affect words after "yourself" in the dictionary, so doesn't really matter
    left_table_dummy = Token('[160]^', '-')
    right_table_dummy = Token('[1240]^', 'zzzzzzz')


    def apply(self, query_token):
        '''apply wordbank if the token is known'''
        if query_token in self._dict:
            t = Token(query_token.raw, plaintext=self._dict[query_token])
            t.source = 'wordbank'
            t.prob = 1
            return t
        else:
            return query_token

    def query_range(self, query_token):
        '''
        Return the tokens on either side of `query_token` or `query_token` itself if it's in the wordbank
        Both tokens are guaranteed to have plaintext that is not None
        '''
        if query_token.ciphertype is None:
            return None

        if query_token.ciphertype == 'table':
            left, right = self._table_tree.find_neighbours(query_token)
        elif query_token.ciphertype == 'dict':
            left, right = self._dict_tree.find_neighbours(query_token)

        # error handling for beginning or end of string
        if left is None and right is None:
            raise ValueError('no tokens have been inserted')
        elif left is None:
            if query_token.ciphertype == 'table':
                left = Wordbank.left_table_dummy
            else:
                left = Wordbank.left_dict_dummy
            right = right.token
        elif right is None:
            if query_token.ciphertype == 'table':
                right = Wordbank.right_table_dummy
            else:
                right = Wordbank.right_dict_dummy
            left = left.token
        else:
            left = left.token
            right = right.token

        return left, right

    def interpolate(self, query_token, vocab):
        '''
        Get a best guess at the contents of query_token
        Return a copy of the token 
        '''

        if query_token.ciphertype is None:
            return query_token
        else:
            left, right = self.query_range(query_token)
            interpolated_relative_position = (int(query_token) - int(left)) / (int(right) - int(left)) # scaled 0 .. 1

            modern_left = vocab.lookup_word(left.plaintext, fuzzy=True)
            modern_right = vocab.lookup_word(right.plaintext, fuzzy=True, left=False)

            interpolated_position = round(modern_left + interpolated_relative_position * (modern_right - modern_left))
            interpolated_plaintext = vocab.lookup_id(interpolated_position)

            return Token(query_token.raw, plaintext=interpolated_plaintext)