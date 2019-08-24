# functions for deciphering text

import re
import bisect
import collections
import copy

import numpy as np
import scipy

from wordbank import Token, Wordbank

vocab = inv_vocab = None # modern dictionary. these are globals and don't change

def basic_substitute(ciphertext, wordbank):
    '''
    replace strings in ciphertext with known decryptions from wordbank

    ciphertext: string
    wordbank: instance of Wordbank
    '''

    token_list = tokenize_ciphertext(ciphertext)
    return ' '.join(token.plaintext for token in token_list)

def tokenize_ciphertext(ciphertext):
    '''
    return a token_list
    '''
    return list(map(Token, re.split(r'\s', ciphertext)))

# def apply_wordbank(token_list, wordbank):
#     for token in token_list:
#         if token in wordbank:
#             token.plaintext = wordbank[token]

def load_vocab(filename):
    '''
    load a vocabulary. 

    input should be a sorted list
    '''
    _vocab = {} # word -> index
    _inv_vocab = [] # index -> word
    with open(filename) as f:
        for idx, line in enumerate(f):
            _vocab = {line: idx}
            _inv_vocab.append(line)

    _vocab['<unk>'] = len(_vocab)
    _inv_vocab.append('<unk>')

    return _vocab, _inv_vocab

# noisy channel model
def lm(token_list):
    '''
    read a context and return one-hot predictions over the output space

    output: 
    '''

    # PLACEHOLDER UNTIL I GET A REAL LM
    # fake language model that simply uses the length. shorter words are better.

    score = 0

    # these are constant for unigrams
    likelihood = np.array([1 / len(word) for word in inv_vocab])
    logprob = np.log(likelihood / likelihood.sum())

    # get score over history
    for token in token_list:
        score += vocab[token.plaintext]

    # add history score to score for each possible addition
    return score + logprob


def prior(token, wordbank):
    '''
    unigram log-probability of each token given its approximate location. returns an array of the same shape as lm
    model P(decoding|encoding) with a beta distribution (because it's normalized 0 to 1)
    '''

    left, right = wordbank.query_range(token)

    # find boundary words in modern vocab
    modern_left = bisect.bisect_left(modern_dict, left.plaintext)
    modern_right = bisect.bisect(modern_dict, right.plaintext)

    # set parameters of the beta distribution with mean where the word is relative to its anchor points
    mean = (int(token) - int(left)) / (int(right) - int(left))
    variance = 0.1 
    n = mean * (1 - mean) / variance

    dist_params = {
        a: mean * n,
        b: (1 - mean) * n,
        loc: modern_left, # shift and scale to be about [0,1), and have variance 1
        scale: modern_right - modern_left,
    }
    if dist_params['a'] < 1 or dist_params['b'] < 1:
        raise ValueError('a, b must be > 1 to have a unimodal distribution')

    # probability of each word is the integral of the PDF from 
    result = np.zeros(10000,)
    for word_idx in range(modern_left, modern_right):
        result[vocab[word]] = scipy.stats.beta.logcdf(word_idx + 1, **dist_params) - scipy.stats.beta.logcdf(word_idx, **dist_params)
    return result

Candidate = collections.namedtuple('Candidate', ['logprob', 'tokens'])

def decode(token_list, wordbank, beam_size=16):
    '''beam search decoding'''

    beam = [Candidate(logprob=0, tokens=[])]

    for token in token_list:
        new_candidates = []

        for beam_idx, candidate in enumerate(beam):
            if token.ciphertype is None or token.plaintext != '<unk>':
                # probably a hard-coded string. doesn't affect our model's probability
                candidate.tokens.append(token)
                new_candidates.append(candidate)

            elif token.ciphertype == 'dict':
                # prior is a beta distribution over the bounds established by the wordbank table

                lm_logprob = lm(candidate.tokens)
                prior_logprob = prior(wordbank, tokens)

                logprob = lm_logprob + prior_logprob
                
            elif token.ciphertype == 'table':
                # prior is a uniform distribution over all english words
                logprob = lm(candidate.tokens)

            # update beam
            if token.ciphertype == 'dict' or token.ciphertype == 'table':
                # get k best extensions to the current sequence and add them to the beam
                k = (beam_size - beam_idx)
                best_extensions = np.argpartition(logprob, len(vocab) - np.arange(k))[-k:] # np array of ints, shape: (k,)

                for ext_idx in best_extensions:
                    new_token = copy.copy(token)
                    new_token.plaintext = inv_vocab[ext_idx]
                    new_candidates.append(Candidate(logprob, candidate.tokens + [new_token]))

        beam = sorted(new_candidates)[-beam_size:] # todo: could save log(beam) time here by doing a partial sort only
        print(beam)
        
    print(beam)
    return beam[-1]

if __name__ == '__main__':
    wordbank = Wordbank()
    wordbank.load('wordbank.without.2880')

    vocab, inv_vocab = load_vocab('dict.modern')
    ciphertext = '[556]^ 586[26]- Ferdinand 2 [678]^ 95 ? [1235]^ y ? [433]^ [79]^ [664]^'
    print(decode(tokenize_ciphertext(ciphertext), wordbank))