from bisect import bisect_left, bisect_right

class Vocab():
    '''Container for a complete vocabulary'''

    def __init__(self, filename):
        self.filename = filename

        self._vocab = {} # word -> index
        self._inv_vocab = [] # index -> word, should be sorted lexographically
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

        self._vocab['<unk>'] = len(self._vocab)
        self._inv_vocab.append('<unk>')

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
            return self._vocab['<unk>']
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

    @property
    def words(self):
        '''Return a list of words'''
        return self._inv_vocab

    def __len__(self):
        return len(self._inv_vocab)