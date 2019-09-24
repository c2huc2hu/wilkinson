# These should return lists of tokens

import re
from collections import Counter

from wordbank import Token

def tokenize_ciphertext(ciphertext):
    '''
    return a token_list
    '''
    result = []
    for raw_token in re.split(r' |({})|({})'.format(
            r'\d+\.\[\d+\][=-]', # re that picks up things using the dict cipher
            r'\[\d+\]\^' # re that picks up things using the table cipher
        ), ciphertext):
        if raw_token:
            result.append(Token(raw_token))

    return result

def add_frequency_attack(wordbank, ciphertext):
    # add a unigram frequency attack to the wordbank
    # greedily assign frequently used (count >= 3) tokens to any matching word in the most common 100 english words

    with open('wordbanks/english-words') as fh:
        frequent_english_words = [line.strip().lower() for line in fh]

    # remove known words from the list of frequent_english_words
    for word in wordbank.words:
        try:
            frequent_english_words.remove(word)
        except ValueError:
            pass

    counter = Counter(ciphertext)
    for token, count in counter.most_common():
        if count <= 5:
            break # only consider words with enough data to be sure about them


        # token is not solvable with alphabetic methods, and probably represents a proper noun
        # OR token is already solved, don't need to solve it
        if (token.ciphertype != 'table' and token.ciphertype != 'dict'
            or not token.is_unk()):
                continue

        left, right = wordbank.query_range(token)
        print(left, right, repr(token))
        for word in frequent_english_words:
            if left.plaintext < word < right.plaintext:
                print('chose word', word)
                wordbank.add(token.raw, word, 'frequency_attack')
                frequent_english_words.remove(word)
                break


from token_lattice import TokenLattice, LengthLanguageModel
from language_models.beam_search import beam_search
from language_models.unigram import UnigramLanguageModel, BigramLanguageModel, SlowGPTLanguageModel

def beam_search_pass(ciphertext, wordbank, alpha=1, beam_width=8):
    '''
    Use a language model and do beam search
    Modifies ciphertext, and returns a reference to it

    ciphertext - a list of tokens. these tokens are modified
    return - a list of tokens, i.e. `ciphertext` but modified

    '''

    lattice = TokenLattice(wordbank)
    # lm = LengthLanguageModel(wordbank.vocab)
    # lm = UnigramLanguageModel(wordbank.vocab)
    lm = SlowGPTLanguageModel()

    beams = beam_search(ciphertext, lm, lattice, beam_width, alpha)
    best_result = beams[0]

    # join tokens to decoded text
    for token, word in zip(ciphertext, best_result):
        token.plaintext = word
    return ciphertext

def dump_lattice(ciphertext, wordbank):
    lattice = TokenLattice(wordbank)
    lattice.dump_carmel_lattice(ciphertext, 'output/ciphertext.lattice')
    print('done dumping')