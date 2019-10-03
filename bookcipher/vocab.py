import itertools
from bisect import bisect_left, bisect_right

from nltk.corpus import wordnet as wn # pattern's wordnet is just a wrapper around nltk, and doesn't support getting all POS
from pattern.en import wordnet, pluralize, lexeme

# python 3.7 hack to initialize pattern: https://github.com/clips/pattern/issues/243#issuecomment-428328754
try: lexeme('walk')
except: pass


# override some common words to reduce branching factor
exceptions = {
    'a': ['a', 'an'],
    'are': ['are'],
    'be': ['be', 'being', 'been'],
    'being': ['be', 'being', 'been'],
    'been': ['be', 'being', 'been'],
    'is': ['is'],
    'was': ['was', 'were'],
    'were': ['was', 'were'],
    'have': ['have', 'has', 'had', 'having'],
    'having': ['have', 'has', 'had', 'having'],

    # not inflected forms, but corrections to enable
    'on': ['on', 'in'], # frequently see "in" where "on" should be, e.g. pg. 2880. this distinction can be hard for non-native speakers
}

def inflect(word):
    '''
    word: a string
    return a list of inflected forms of a word
    '''

    if word in exceptions:
        return exceptions[word]

    # get all possible parts of speech for a word from wordnet
    synsets = wn.synsets(word)
    pos = {lemma.pos() for lemma in synsets}

    # decline/ conjugate word
    result = {word}

    if 'n' in pos:
        result.add(pluralize(word))
    if 'v' in pos:
        # get all forms of the verb, e.g. lexeme('be') -> ['be', 'am', 'was', ... 'weren\'t']
        # but it's better to be overgeneral
        result.update(lexeme(word)) 

    return list(result)


class Vocab():
    '''Container for a complete vocabulary'''

    def __init__(self, filename=None):
        self.filename = filename

        self._vocab = {} # word -> index
        self._inv_vocab = [] # index -> word, should be sorted lexographically
        self._inflected_inv_vocab = [] # index -> inflected forms of word
        self._flat_inflected_vocab = []

        if filename:
            self.load(self.filename)

    def load(self, filename):
        '''
        load a vocabulary.

        input should be a sorted list
        '''

        if len(self._vocab):
            raise ValueError('can\'t call load twice')

        with open(filename) as f:
            for idx, line in enumerate(f):
                line = line.strip()
                self._vocab[line] = idx
                self._inv_vocab.append(line)

            self._inflected_inv_vocab = [inflect(word) for word in self._vocab]
            self._flat_inflected_vocab = list(itertools.chain.from_iterable(self._inflected_inv_vocab))

        # mapping vocab -> flat_*_vocab, such that flat_inflected_vocab[_flat_index[i]:_flat_index[i+1]] are all versions of the same word
        self._flat_index = list(itertools.accumulate(map(len, self._inflected_inv_vocab)))
        self._flat_index.insert(0, 0)

        self._vocab['{unk}'] = len(self._vocab)
        self._inv_vocab.append('{unk}')


    def lookup_id(self, word):
        return self._inv_vocab[word]

    def lookup_word(self, word, fuzzy=False, left=True):
        '''
        Find words from their position in the dictionary

        If `fuzzy`, either the position to the left or right of the word will be returned if the word is not in the dictionary
        Return an integer
        '''

        if word in self._vocab:
            return self._vocab[word]
        elif not fuzzy:
            return self._vocab['{unk}']
        else: # fuzzy is true
            bisect_fcn = bisect_left if left else bisect_right # which side to err on if the word isn't found
            return bisect_fcn(self._inv_vocab, word)

    def lookup_prefix(self, prefix):
        '''
        Return the range of word indices that start with prefix. 
        Left index is inclusive, right index is not, similar to slicing, i.e. vocab.words[slice(lookup_prefix(prefix))] -> all words that start with prefix

        Example:
        Say vocab = ['a', 'aback', 'abacus', 'abandon', 'abandoned', 'abandonment']
        lookup_range('abacus') -> (2, 3)  # only 'abacus' matches
        lookup_range('abandon') -> (3, 6) # 'abandon' through 'abandonment' match
        '''
        prefix = prefix.lower()
        return bisect_left(self._inv_vocab, prefix), bisect_right(self._inv_vocab, prefix + '\uffff')

    def to_flat_idx(self, idx):
        '''Return a mapping vocab.words[idx] -> vocab.inflected_words[flat_idx]'''
        return self._flat_index[idx]

    @property
    def words(self):
        '''Return a list of words'''
        return self._inv_vocab

    @property
    def inflected_words(self):
        return self._flat_inflected_vocab

    def __len__(self):
        return len(self._inv_vocab)