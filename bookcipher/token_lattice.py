import itertools
from scipy.stats import beta
import numpy as np

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
        })

    def inflect(self, word):
        if word not in self.inflected_vocab:
            self.inflected_vocab[word] = inflect(word)
        return self.inflected_vocab[word]

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
        
        # import pdb; pdb.set_trace()

        # parameterize a beta distribution with mean at the interpolated position and beam 
        b = 1 # parameterizes the sharpness of the distribution.
                 # TODO: actually fit data to figure out what this should be
        mean = interpolated_position
        a = (mean * b) / (1 - mean) # controls where the peak is

        # integrate probability mass
        target_idx = self.vocab.lookup_word(target_word)
        prob = beta.cdf((target_idx + 1 - left_idx) / scale, a=a, b=b) - beta.cdf((target_idx - left_idx) / scale, a=a, b=b)
        return np.log(prob)

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

    # TODO: LRU cache this?
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
            return ((inflected_word, -np.log(len(inflected_forms))) for inflected_word in inflected_forms)
        else:
            # return everything in the vocab between the anchor points including inflected forms
            # note that we distribute probability mass equally over each of the inflected forms
            # this penalizes things with more forms (esp. irregular verbs)
            left_idx, right_idx, _ = self.wordbank.get_anchors(source_token)

            result = [(inflected_word, self.backward_probs(source_token, raw_word) / len(self.inflect(raw_word)))
                for raw_word in self.vocab.words[left_idx:right_idx]
                for inflected_word in self.inflect(raw_word)]

            # print('branching from {} with possibilities: {}'.format(repr(source_token), [x[0] for x in result]))
            return result


class TokenLanguageModel(LanguageModel):
    def __init__(self, vocab):
        '''
        vocab: pre-initialized instance of Vocab
        '''
        self.vocab = vocab.words # index -> word
        self.vocab_indices = vocab._vocab # word -> index

    def initialize(self, source):
        pass

    def next_token(self, s):
        '''return log probabilites over the vocab'''

        # TODO: swap this for a real language model. this just penalizes length
        return [-len(x) for x in self.vocab]

class UnigramLanguageModel(LanguageModel):
    # todo: brown corpus unigram frequencies OR project gutenberg frequencies from nltk
    pass


if __name__ == '__main__':
    from passes import tokenize_ciphertext
    from vocab import Vocab
    from wordbank import Wordbank

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

    lm = TokenLanguageModel(v)
    lattice = TokenLattice(wb)

    ct = [wb.apply(tok) for tok in ct]
    print(ct)
    print('Best predictions (in order)', beam_search(ct, lm, lattice, alpha=1, beam_width=32))