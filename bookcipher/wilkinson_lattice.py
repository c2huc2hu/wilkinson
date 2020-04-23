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
    def __init__(self, label, log_prob, token=None, uninflected_form=None):
        super().__init__(label, log_prob)
        self.token = token
        if uninflected_form is None:
            self.uninflected_form = self.label
        else:
            self.uninflected_form = uninflected_form

class WilkinsonLattice(Lattice):
    def __init__(self, source, wordbank, beta, small_vocab=None):
        self.source = source
        self.wordbank = wordbank
        self.vocab = wordbank.vocab
        self.beta = beta

        if small_vocab is None:
            self.small_vocab = self.vocab
        else:
            self.small_vocab = small_vocab

        # self.small_vocab = self.vocab

        # parse lattice into general Lattice format
        self.lattice = defaultdict(lambda: defaultdict(list))

        i = 0
        for token in source:
            if not token.plaintext.isspace(): # don't load space tokens
                self.lattice[i][i+1] = self._batch_probs(token)

                # re-normalize probabilities
                sum_ = sum(np.exp(edge.log_prob) for edge in self.lattice[i][i+1])
                for edge in self.lattice[i][i+1]:
                    edge.log_prob -= np.log(sum_)

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
            else: # condition over the smaller vocab, because the table cipher has common words
                return [WilkinsonLatticeEdge(word, 0, token=source_token) for word in self.small_vocab.inflected_words if word.endswith(source_token.suffix)]
        # no point in assigning probabilities to known words
        elif not source_token.is_unk(self.wordbank):
            inflected_forms = inflect(source_token.plaintext)
            return [WilkinsonLatticeEdge(form, 0, token=source_token, uninflected_form=source_token.plaintext) for form in inflected_forms if form.endswith(source_token.suffix)]

        # else: # interpolate position

        # Positions in a modern dictionary
        # Can scale things to (x - anchor_left) / scale to scale it from 0 to 1
        anchor_left, anchor_right, mean = self.wordbank.get_anchors(source_token)
        anchor_right += 1 # include endpoints to account for repeated words
        scale = anchor_right - anchor_left
        if scale == 0:
            raise ValueError('Found a token that can be deduced from wordbank. Did you forget to call wordbank.apply?')

        # Parameterize distribution
        b = self.beta # parameterizes the sharpness of the distribution.
              # TODO: actually fit data to figure out what this should be
        a = (mean * b - 2 * mean + 1) / (1 - mean) # controls mode of the distribution

        # distribute probabilities to `scale` probability buckets, corresponding to tokens in vocab
        cdf = beta.cdf(np.arange(scale + 1) / scale, a=a, b=b)
        probability_buckets = np.log(cdf[1:] - cdf[:-1])

        result = []
        for raw_word, inflected_forms, log_prob in zip(self.vocab.words[anchor_left:anchor_right], self.vocab._inflected_inv_vocab[anchor_left:anchor_right], probability_buckets):
            # print(f'Distributing {prob} probability mass from word {raw_word} to {len(inflected_forms)} buckets ({prob - productive_penalty})')
            # print('Forms: ')

            # Skip words with extremely low probability to save work in the LM
            if log_prob < -50:
                continue

            for form in inflected_forms:
                # print(form, end=', ')
                if form.endswith(source_token.suffix):
                    result.append(WilkinsonLatticeEdge(form, log_prob, token=source_token, uninflected_form=raw_word))
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
    from passes import tokenize_ciphertext
    from wordbank import Wordbank
    from vocab import Vocab
    vocab = Vocab('dict.modern')
    wb = Wordbank(vocab)
    wb.load('wordbanks/wordbank.clean2')
    ciphertext = tokenize_ciphertext('[664]^ [526]^ [629]^ [1078]^ [752]^ [1216]^ 192.[10]- [172]^ [177]^ [782]^', ) # first line of the test set
    lattice = WilkinsonLattice(ciphertext, wb, 5, vocab)
    lattice.to_carmel_lattice('lattices/wilkinson_head.lattice')

