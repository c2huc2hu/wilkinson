import itertools
from scipy.stats import beta
import numpy as np
from functools import lru_cache

from nltk.corpus import wordnet as wn # pattern's wordnet is just a wrapper around nltk, and doesn't support getting all POS
from pattern.en import wordnet, pluralize, lexeme

# python 3.7 hack to initialize pattern: https://github.com/clips/pattern/issues/243#issuecomment-428328754
try: lexeme('walk')
except: pass


from language_models.lattices import Lattice
from language_models.beam_search import LanguageModel, beam_search

def inflect(word):
    '''
    word: a string
    return a list of inflected forms of a word as
    '''

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

class TokenLattice(Lattice):
    def __init__(self, wordbank):
        self.wordbank = wordbank
        self.vocab = wordbank.vocab

        # precompute inflected forms of every word in the vocab
        self.inflected_vocab = {word: inflect(word) for word in self.vocab.words}

        # override some common words to reduce branching factor
        self.inflected_vocab.update({
            'a': ['a'],
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
            'on': ['on', 'in'], # frequently see "in" where "on" should be, e.g. pg. 2880. this distinction can be hard
        })

    def inflect(self, word):
        if word not in self.inflected_vocab:
            self.inflected_vocab[word] = inflect(word)
        return self.inflected_vocab[word]

    @lru_cache(512)
    def backward_probs(self, source_token, target_word):
        '''
        Return log(p(target_word|source_token))
        Probabilities do not sum to 1

        source_token: a Token
        target_word: a word

        return a float <=0
        '''

        # unknown words aren't decipherable
        if source_token.ciphertype is None:
            return 0
        # no point in assigning probabilities to known words. could assign probabilities over possible inflections
        if not source_token.is_unk(self.wordbank):
            return 0
        
        left_idx, right_idx, interpolated_position = self.wordbank.get_anchors(source_token)

        # scale things
        scale = right_idx - left_idx
        if scale == 0:
            raise ValueError('Found a token that can be deduced from wordbank. Did you forget to call wordbank.apply?')
        
        # parameterize a beta distribution with mean at the interpolated position and beam 
        b = 1 # parameterizes the sharpness of the distribution.
              # TODO: actually fit data to figure out what this should be
        mean = interpolated_position
        a = (mean * b) / (1 - mean) # controls where the peak is

        # integrate probability mass
        target_idx = self.vocab.lookup_word(target_word)

        prob = beta.cdf((target_idx + 1 - left_idx) / scale, a=a, b=b) - beta.cdf((target_idx - left_idx) / scale, a=a, b=b)

        return np.log(prob) # np.log(0) warnings here are okay, because it just indicates very unlikely words

    def possible_substitutions(self, source_token):
        '''
        Return a list of plaintext substitutions for source_token

        source_token: a Token
        return a list of strings
        '''

        if source_token.ciphertype is None:
            return source_token.plaintext
        elif not source_token.is_unk():
            # return inflected forms
            return self.inflected_vocab[source_token.plaintext]
        else:
            # return everything in the vocab between the anchor points including inflected forms
            left_idx, right_idx, _ = self.wordbank.get_anchors(source_token)
            return itertools.chain(*self.inflected_vocab[left_idx:right_idx])

    @lru_cache(2) # since we process token-by-token, the LRU cache can be size 1
    def possible_substitutions_and_probs(self, source_token):
        '''
        Return tuples (word, probability) to distribute probability mass over each inflected form
        This is slightly more efficient and prevents us from having to canonical-ize source_token

        return: a generator that yields elements (word: string, log_probability: float)
        '''

        if source_token.ciphertype is None:
            return ((source_token.plaintext, 0),)
        elif not source_token.is_unk():
            # return inflected forms
            inflected_forms = self.inflect(source_token.plaintext)
            # print('inflecting {} with possibilities: {}'.format(repr(source_token), inflected_forms))
            return [(inflected_word, -np.log(len(inflected_forms))) for inflected_word in inflected_forms]
        else:
            # return everything in the vocab between the anchor points including inflected forms
            # note that we distribute probability mass equally over each of the inflected forms
            # this penalizes things with more forms (esp. irregular verbs)
            left_idx, right_idx, _ = self.wordbank.get_anchors(source_token)

            # for raw_word in self.vocab.words[left_idx:right_idx]:
            #     print(self.inflect(raw_word))

            result = [(inflected_word, self.backward_probs(source_token, raw_word) - np.log(len(self.inflect(raw_word))))
                for raw_word in self.vocab.words[left_idx:right_idx]
                for inflected_word in self.inflect(raw_word)]

            # print('branching from {} with possibilities: {}'.format(repr(source_token), [x[0] for x in result]))
            return result


    def integrate_prob(self, prefix, source_token):
        '''
        Get log probability mass $$\\sum_{word starts with prefix} P(prefix | token)$$,
        i.e. the sum of everything in `self.wordbank.vocab` for all words starts with `prefix`

        If `prefix` isn't in the dictionary, give it 1/2 the probability mass of where it would be found

        Note: probabilities won't sum to 1, because GPT's vocab >> the dictionary's vocab

        `prefix`: a string

        return: a float
        '''

        # known tokens are added explicitly ahead of the beam search
        if source_token.ciphertype is None:
            return -np.inf

        prefix = prefix.lower()

        # assign the word probability based on where it is in the dictionary
        prefix_left, prefix_right = self.vocab.lookup_prefix(prefix)

        # prefix is not in dictionary, will assign it half the probabilty mass of the space it occupies, centred around where it should be
        if prefix_left == prefix_right:
            prefix_left, prefix_right = prefix_left + 1/4, prefix_left + 3/4

        # find anchor points, i.e. the neighbours of source_token in the wilkinson dictionary
        anchor_left, anchor_right, interpolated_position = self.wordbank.get_anchors(source_token)    
        scale = anchor_right - anchor_left
        if scale == 0:
            raise ValueError('Found a token that can be deduced from wordbank. Did you forget to call wordbank.apply?')

        # parameterize a beta distribution with mean at the interpolated position and beam 
        b = 1 # parameterizes the sharpness of the distribution.
              # TODO: actually fit data to figure out what this should be
        mean = interpolated_position
        a = (mean * b) / (1 - mean) # controls where the peak is

        # give probability mass
        prob = beta.cdf((prefix_right - anchor_left) / scale, a=a, b=b) - beta.cdf((prefix_left - anchor_left) / scale, a=a, b=b) # todo: vectorize this
        lattice_prob = np.log(prob)
        return lattice_prob # log prob

    def dump_carmel_lattice(self, source, filename):
        '''
        dump a carmel lattice to a file
    
        e.g. 
        0 (1 "a" 0.01)
        0 (1 "b" 0.02)
        '''
        with open(filename, 'w') as fh:
            print(len(source), file=fh)
            for i, token in enumerate(source):
                if not token.plaintext.strip():
                    continue  # skip blank tokens

                for next_word, lattice_prob in self.possible_substitutions_and_probs(token):
                    print('({} ({} "{}" {}))'.format(i, i + 1, next_word, np.exp(lattice_prob)), file=fh)


class LengthLanguageModel(LanguageModel):
    def __init__(self, vocab):
        '''
        vocab: pre-initialized instance of Vocab
        '''
        self.vocab = vocab.words # index -> word
        self.vocab_indices = vocab._vocab # word -> index
        self.unk = 0

    def initialize(self, source):
        pass

    def next_token(self, s):
        '''return log probabilites over the vocab'''

        return [-len(x) for x in self.vocab]


if __name__ == '__main__':
    from passes import tokenize_ciphertext
    from vocab import Vocab
    from wordbank import Wordbank
    from language_models.unigram import SlowGPTLanguageModel

    # load data
    v = Vocab('wordbanks/vocab.toy')
    wb = Wordbank(v)
    wb.load('wordbanks/wordbank.toy')
    ct = tokenize_ciphertext('[160]^ [330]^ [960]^ [1078]^ [490]^') # given the toy vocab, should give "a cat sees the egg"

    # test cases
    left, right = wb.query_range(tokenize_ciphertext('[160]^')[0])
    assert left.plaintext == right.plaintext == 'a'

    left, right = wb.query_range(tokenize_ciphertext('[161]^')[0])
    assert left.plaintext == 'a' and right.plaintext == 'day'

    # lm = LengthLanguageModel(v)
    lm = SlowGPTLanguageModel(v)
    lattice = TokenLattice(wb)

    ct = [wb.apply(tok) for tok in ct]
    print(ct)
    print('Best predictions (in order)', beam_search(ct, lm, lattice, alpha=1, beam_width=256))