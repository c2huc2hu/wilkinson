# These should return lists of tokens

import re
from collections import Counter

from wordbank import Token

def apply_literals(ciphertext, filename):
    '''
    Do replacements that aren't part of the alphabetic wordbank
    Replace known groups of characters, e.g. [1235]^ y -> my
    and extra stuff before or after the table

    Also strip out inflection markers e.g. +ing

    Takes a string, returns a string
    '''

    # Replace all literals
    with open(filename) as fh:
        for line in fh:
            if line.startswith('#') or line.isspace():
                continue # skip comments and blank lines

            raw, source, plaintext = line.split('\t')

            ciphertext = ciphertext.replace(raw.strip(), plaintext)

    # # Delete unused inflection markers
    # ciphertext = re.sub(r'\+\w+', '', ciphertext)

    return ciphertext

split_re = re.compile(r'''
    \s
    |(\d+\.\[\d+\][=-]) # dict cipher
    |(\[\d+\]\^)  # table cipher
    |(\s\.)       # separate out periods
''', flags=re.VERBOSE)
def split_ciphertext(ciphertext):
    return [match for match in re.split(split_re, ciphertext) if match]

def tokenize_ciphertext(ciphertext):
    '''
    return a token_list
    '''

    tokens = []
    for token in split_ciphertext(ciphertext):
        if token.startswith('+'):
            tokens[-1].suffix = token[1:] # drop leading +
        else:
            t = Token(token)
            tokens.append(t)

    return tokens

def visualize(history):
    result = []
    for beam in history[1:]: # first element contains empty beam

        # wilkinson specific properties
        token = beam.lattice_edge.token
        uninflected_form = beam.lattice_edge.uninflected_form
        new_word = beam.lattice_edge.label

        style_dict = {
            'literal': 'turquoise',
            'clean_wordbank': 'lightblue',
            'miro': 'skyblue',
            '2880': 'lightskyblue',
            'guess': 'limegreen',
            'context': 'gold',
            'guessed_proper_noun': 'pink',
        }

        if token.source in style_dict:
            style = 'background-color:' + style_dict[token.source] + ';'
        else:
            style = ''

        result.append(
            '''
            <div class="tooltip" style="{style}">
                {token}
                {plaintext}
                <span class="tooltiptext">
                    uninflected:{uninflected_form}<br>
                    lmprob:{lm_prob:.2}<br>
                    latticeprob:{lattice_prob:.2}<br>
                    source:{source}
                </span>
            </div>
            '''
        .format(plaintext=new_word.replace('<', '&lt;'),
                 token=token.raw,
                 source=token.source,
                 lm_prob=float(beam.lm_prob),
                 lattice_prob=float(beam.lattice_prob),
                 style=style,
                 uninflected_form=uninflected_form,
        ))
    return '\n'.join(result)


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


from token_lattice import TokenLattice # , LengthLanguageModel
# from language_models.beam_search import beam_search
# from language_models.unigram import UnigramLanguageModel, BigramLanguageModel, SlowGPTLanguageModel

from wordpiece_beamsearch import beam_search, GPTLanguageModel

def beam_search_pass(ciphertext, wordbank, alpha=1, beam_width=8):
    lattice = TokenLattice(wordbank)
    lm = GPTLanguageModel()

    beams = beam_search(ciphertext, lm, lattice, beam_width, alpha)
    best_result = beams[0]
    return best_result

# def beam_search_pass(ciphertext, wordbank, alpha=1, beam_width=8):
#     '''
#     Use a language model and do beam search
#     Modifies ciphertext, and returns a reference to it

#     ciphertext - a list of tokens. these tokens are modified
#     return - a list of tokens, i.e. `ciphertext` but modified

#     '''

#     lattice = TokenLattice(wordbank)
#     # lm = LengthLanguageModel(wordbank.vocab)
#     # lm = UnigramLanguageModel(wordbank.vocab)
#     lm = SlowGPTLanguageModel()

#     beams = beam_search(ciphertext, lm, lattice, beam_width, alpha)
#     best_result = beams[0]

#     # join tokens to decoded text
#     for token, word in zip(ciphertext, best_result):
#         token.plaintext = word
#     return ciphertext
