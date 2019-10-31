from collections import defaultdict
import numpy as np
from scipy.stats import beta
from general_beamsearch import Lattice, LatticeEdge
from vocab import inflect

class WilkinsonLatticeEdge(LatticeEdge):
    '''
    Wrap LatticeEdge in a class to add extra attributes without breaking
    backward compatibility. This is definitely a hack.
    '''
    def __init__(self, label, prob, token=None, uninflected_form=None):
        super().__init__(label, prob)
        self.token = token
        if uninflected_form is None:
            self.uninflected_form = self.label
        else:
            self.uninflected_form = uninflected_form

class WilkinsonLattice(Lattice):
    def __init__(self, source, wordbank):
        self.source = source
        self.wordbank = wordbank
        self.vocab = wordbank.vocab

        # parse lattice into general Lattice format
        self.lattice = defaultdict(lambda: defaultdict(list))

        i = 0
        for token in source:
            if not token.plaintext.isspace(): # don't load space tokens
                self.lattice[i][i+1] = self._batch_probs(token)
                i += 1

        self.start_state = 0
        self.final_state = i

    def _batch_probs(self, source_token):
        '''
        Return a list of pairs (word, lattice_probability)
        '''

        # unknown words aren't decipherable
        if source_token.ciphertype is None:
            # TODO: inflect depending on plaintext
            if source_token.plaintext.isspace():
                return [WilkinsonLatticeEdge('', 0, token=source_token)] # ignore empty tokens
            elif '[' not in source_token.plaintext:
                return [WilkinsonLatticeEdge(source_token.plaintext, 0, token=source_token)] # return unchanged plaintext for literals
            else: # condition over the whole output space
                return [WilkinsonLatticeEdge(word, -np.log(len(self.vocab.inflected_words)), token=source_token) for word in self.vocab.inflected_words]
        # no point in assigning probabilities to known words
        elif not source_token.is_unk(self.wordbank):
            inflected_forms = inflect(source_token.plaintext)
            productive_penalty = np.log(len(inflected_forms)) # distribute probability over all inflected forms
            return [WilkinsonLatticeEdge(form, -productive_penalty, token=source_token, uninflected_form=source_token.plaintext) for form in inflected_forms]

        # else: # interpolate position

        # Positions in a modern dictionary
        # Can scale things to (x - anchor_left) / scale to scale it from 0 to 1
        anchor_left, anchor_right, mean = self.wordbank.get_anchors(source_token)
        scale = anchor_right - anchor_left
        if scale == 0:
            raise ValueError('Found a token that can be deduced from wordbank. Did you forget to call wordbank.apply?')

        # Parameterize distribution
        b = 1 # parameterizes the sharpness of the distribution.
              # TODO: actually fit data to figure out what this should be
        a = (mean * b) / (1 - mean) # controls where the peak is

        # distribute probabilities to `scale` probability buckets, corresponding to tokens in vocab
        cdf = beta.cdf(np.arange(scale + 1) / scale, a=a, b=b)
        probability_buckets = np.log(cdf[1:] - cdf[:-1])

        result = []
        for raw_word, inflected_forms, prob in zip(self.vocab.words[anchor_left+1:anchor_right], self.vocab._inflected_inv_vocab[anchor_left+1:anchor_right], probability_buckets):
            productive_penalty = np.log(len(inflected_forms)) # distribute probability over all inflected forms
            # print(f'Distributing {prob} probability mass from word {raw_word} to {len(inflected_forms)} buckets ({prob - productive_penalty})')
            # print('Forms: ')
            for form in inflected_forms:
                # print(form, end=', ')
                result.append(WilkinsonLatticeEdge(form, prob - productive_penalty, token=source_token, uninflected_form=raw_word))
            # print('')

        return result

class NoSubstitutionLattice(Lattice):
    def __init__(self, source):
        self.source = source

        # parse lattice into general Lattice format
        self.lattice = defaultdict(lambda: defaultdict(list))

        i = 0
        for token in source:
            if not token.plaintext.isspace(): # don't load space tokens
                self.lattice[i][i+1] = self._batch_probs(token)
                i += 1

        self.start_state = 0
        self.final_state = i

    def _batch_probs(self, source_token):
        '''
        Return a list of LatticeEdges (word, lattice_probability), only using what's provided 
        '''

        if source_token.plaintext == 'i':
            return [LatticeEdge('I', 0)]
        elif source_token.plaintext.isspace():
            return [LatticeEdge('', 0)]
        else:
            return [LatticeEdge(source_token.plaintext, 0)]


if __name__ == '__main__':
    lm = GPTLanguageModel(vocab)
    lattice = TokenLattice(wordbank)

