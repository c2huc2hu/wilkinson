# functions for handling wordbanks

import re
from functools import total_ordering, lru_cache

from avltree import TokenAVL

MIN_WORD = 1 # index of first word in table cipher
MAX_WORD = 82672 # maximum word in table cipher


@total_ordering
class Token():
    dict_re = r'(?P<page>\d+)\.\[(?P<row>\d+)\](?P<col>[=-])'
    table_re = r'\[(?P<row>\d+)\]\^'

    def __init__(self, raw_string, plaintext='<unk>', source=''):
        self.raw = raw_string.strip()
        self.plaintext = plaintext
        self.ciphertype = None # which type of cipher this uses. one of: None (unparsed), dict or table

        self.source = source
        self.prob = 0
        self.suffix = '' # suffixes, e.g. +ing

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

            if self.row < MIN_WORD: # table of proper nouns, names or places
                self.ciphertype = None
                self.plaintext = 'America'
                self.source = 'guessed_proper_noun'
                self.prob = 1

            if self.row > MAX_WORD: # words added afterwards
                self.ciphertype = None
                del self.row
                self.plaintext = 'America'

        else:
            self.plaintext = self.raw
            self.source = 'literal'
            self.prob = 1

    def is_unk(self, wb=None):
        '''
        Return whether this token object has plaintext assigned to it. If wb is provided, checks whether this token is known by that wordbank

        wb: an instance of wordbank
        '''
        if wb and self in wb._dict:
            return False
        else:
            return self.plaintext == '<unk>'

    def write(self):
        if self.ciphertype is None:
            return self.raw
        elif self.ciphertype == 'table':
            self.raw = '[{}]^'.format(self.row)
            return self.raw
        elif self.ciphertype == 'dict':
            self.raw = '{}.[{}]'.format(self.page, self.row) + ('=' if self.col == 1 else '-')
            return self.raw

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

            # key mapping source -> html colors
            source_key = {
                'literal': 'turquoise',
                'clean_wordbank': 'lightblue',
                'miro': 'skyblue',
                '2880': 'lightskyblue',
                'guess': 'limegreen',
                'frequency_attack': 'gold',
                # 'interpolate': 'darkgreen',
            }

            if self.source in source_key:
                style = 'background-color:' + source_key[self.source] + ';'
            else:
                style = 'background-color: rgb(0,{},0);'.format(127 + self.prob*128)

            return '''
                <div class="tooltip" style="{style}">{plaintext}
                    <span class="tooltiptext">
                        raw:{raw}<br>p:{prob:.2}<br>source:{source}
                    </span>
                </div>
            '''.format(plaintext=self.plaintext.replace('<', '&lt;'), 
                    source=self.source, raw=self.raw, prob=float(self.prob), style=style)

class Wordbank():
    def __init__(self, vocab):
        self._dict = {} # mapping Token -> word
        self._wordbank_source = {} # mapping Token -> source for debugging, check where every word came from
        self._dict_tree = TokenAVL() # binary search tree storing tokens for those encypted with the dictionary method. used to keep track of ranges
        self._table_tree = TokenAVL() # same as above for the table method
        self.vocab = vocab

    def load(self, filename, wordbank_name=''):
        if wordbank_name and not wordbank_name.endswith('/'):
            wordbank_name += '/'

        with open(filename) as f:
            for line in f:
                if line.startswith('#'): # comment blank lines with #
                    continue
                elif not line.strip(): # skip blank lines
                    continue
                else:
                    location, source, word = line.strip().split('\t')
                    self.add(location, word, source)

    def save(self, filename):
        '''Dump wordbank to file'''

        with open(filename, 'w') as f:
            tokens = list(self._dict.keys())
            tokens.sort()
            for token in tokens:
                print('\t'.join([token.raw, token.source, token.plaintext]), file=f)

    def add(self, raw, plaintext, source=''):
        '''add a word to the wordbank if it's not already there'''
        t = Token(raw, plaintext)

        # dictionary is one page off for these letters
        if t.ciphertype == 'dict' and (source == '1633' or source == '1652'):
            t.page += 1

        if t in self._dict:
            return

        self._dict[t] = plaintext
        self._wordbank_source[t] = source

        if t.ciphertype is not None:
            try:
                left, right = self.query_range(t)
                if not(int(left) <= int(t) <= int(right)) or not(left.plaintext <= t.plaintext <= right.plaintext):  # "know" and "known" to take up consecutive slots
                    raise ValueError('Error when inserting token {}, out of order'.format(repr(t)))
            except IndexError as e:
                pass # ignore errors from empty tree

        if t.ciphertype == 'table':
            self._table_tree.insert(t)
        elif t.ciphertype == 'dict':
            self._dict_tree.insert(t)


    # imaginary words that would go at the beginning and end of a dictionary
    left_dict_dummy = Token('1.[1]-', '-')
    right_dict_dummy = Token('780.[1]-', 'zzzzzzz')     # the last attested word is 44,312 = yourself. it will only affect words after "yourself" in the dictionary, so doesn't really matter
    left_table_dummy = Token('[{}]^'.format(MIN_WORD), '-')
    right_table_dummy = Token('[{}]^'.format(MAX_WORD), 'zzzzzzz')     # last attested alphabetical word is 1219 = young, after that it becomes a lookup table again

    def apply(self, query_token):
        '''apply wordbank if the token is known'''
        if query_token in self._dict:
            t = Token(query_token.raw, plaintext=self._dict[query_token])
            t.source = self._wordbank_source[query_token]
            t.prob = 1.0
            return t
        else:
            return query_token

    def query_range(self, query_token):
        '''
        Return the tokens on either side of `query_token` that have plaintext that is not None
        If `query_token` itself is in the wordbank, return it twice.
        '''
        if query_token.ciphertype is None:
            raise ValueError('query_token must use the table or dictionary cipher')

        if query_token.ciphertype == 'table':
            left, right = self._table_tree.find_neighbours(query_token)
        elif query_token.ciphertype == 'dict':
            left, right = self._dict_tree.find_neighbours(query_token)

        # error handling for beginning or end of string
        if left is None and right is None:
            raise IndexError('no tokens have been inserted')
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

    @lru_cache()
    def get_anchors(self, query_token):
        '''
        Get the bounds on a modern dictionary and the approximate position of a word.
        '''
        if query_token.ciphertype is None:
            raise ValueError('query token must use the table or dictionary cipher')
        elif query_token in self._dict:
            word_idx = self.vocab.lookup_word(self._dict[query_token])
            return word_idx, word_idx, 0
        else:
            left, right = self.query_range(query_token)
            interpolated_relative_position = (int(query_token) - int(left)) / (int(right) - int(left)) # scaled 0 .. 1

            modern_left = self.vocab.lookup_word(left.plaintext, fuzzy=True)
            modern_right = self.vocab.lookup_word(right.plaintext, fuzzy=True, left=False)

            return modern_left, modern_right, interpolated_relative_position

    def interpolate(self, query_token):
        '''
        Get a best guess at the contents of query_token
        Don't add it to the wordbank.
        Return a copy of the token 
        '''

        if query_token.ciphertype is None:
            return query_token
        else:
            left, right = self.query_range(query_token)
            interpolated_relative_position = (int(query_token) - int(left)) / (int(right) - int(left)) # scaled 0 .. 1

            modern_left = self.vocab.lookup_word(left.plaintext, fuzzy=True)
            modern_right = self.vocab.lookup_word(right.plaintext, fuzzy=True, left=False)

            interpolated_position = round(modern_left + interpolated_relative_position * (modern_right - modern_left))
            interpolated_plaintext = self.vocab.lookup_id(interpolated_position)

            t = Token(query_token.raw, plaintext=interpolated_plaintext, source='interpolate')

            # express certainty depending on how big the range is. 
            # we're more certain about smaller ranges, so scale them closer to 1. this also increases colour contrast
            t.prob = interpolated_relative_position ** (1/3)
            return t

    @property
    def words(self):
        '''Return a generator over all known plaintext words in the wordbank'''
        return self._dict.values()

    def run(self, tokenized_ciphertext, interpolate=False, verbose=False):
        '''
        apply a single substitution pass on the cihpertext, and optionally an interpolation pass.
        call this after updating the wordbank

        interpolate should only be called once, because it substitutes all unknown tokens with <unk>
        vocab must be provided if interpolate is True
        '''

        if verbose:
            print('before running')
            print(tokenized_ciphertext)
            print('================== before running')

        # apply wordbank
        wordbanked_ct = [self.apply(token) for token in tokenized_ciphertext]
        if verbose:
            print('text after applying wordbank')
            print(wordbanked_ct)
            print('================== applied wordbank')

        if not interpolate:
            return wordbanked_ct
        else:
            # apply a basic interpolation to make the text easier to read
            deciphered_ct = [self.interpolate(token) if token.is_unk() else token for token in wordbanked_ct]

            if verbose:
                print('text after applying interpolation')
                print(deciphered_ct)
                print('================== applied interpolation')
                print(' '.join(map(str, deciphered_ct)))

            return deciphered_ct
