from collections import namedtuple
import os

import nltk

from wordbank import Wordbank, Token
from vocab import Vocab
from passes import apply_literals, split_ciphertext, tokenize_ciphertext, visualize

from general_beamsearch import beam_search, LengthLanguageModel, UnigramLanguageModel, Lattice
from wilkinson_lattice import WilkinsonLattice, NoSubstitutionLattice
from confidence_model import confidence_model

# Load config
import argparse
parser = argparse.ArgumentParser(description='Solve a bookcipher')
parser.add_argument('-b', '--beam-width', nargs='?', default=4, help='width of beam search. runtime scales linearly', type=int)
parser.add_argument('--lattice_file', help='path to lattice file')
parser.add_argument('--source_file', metavar='source-file', help='source file to decode')
parser.add_argument('--gold_file', metavar='gold-file', help='reference translation for scoring accuracy')
parser.add_argument('--language-model', '--lm', help='which language model to use', choices=['gpt2', 'gpt2-large', 'unigram', 'length', 'oracle', 'ngram', 'none'])
parser.add_argument('--self-learn', help='enable self-learning', action='store_true')
parser.add_argument('-S', '--substitutions', nargs='?', default=5, help='number of substitutions to make each decoding', type=int)
parser.add_argument('--beta', default=5, help='number of substitutions to make each decoding', type=float) # note: changing this can slightly affect accuracy of even the oracle model because of pruning
parser.add_argument('--confidence_model', help='function to determine which words to add to the wordbank', choices=['left', 'oracle'])
parser.add_argument('--oracle', help='choose the best beam with an oracle (cheating experiment)', action='store_true') # warning: using this with GPT gives lower accuracy results because GPT decodes periods without a leading space keeping them from being separated. to get accurate accuracy results, use a different LM
args = parser.parse_args()

if args.source_file is None and args.lattice_file is None:
    args.source_file = 'data/eval/ciphertext.txt'
    args.gold_file = 'data/eval/plaintext.txt'
if args.source_file is not None and args.lattice_file is not None:
    raise ValueError('Cannot supply both a lattice and source file')
print('Config:', args)

# Runtime

# Collect vocabs
# lemma dictionary
vocab = Vocab('dict.modern')

# smaller dictionary (~1000 words) to speed up words outside table
small_vocab = Vocab('wordbanks/vocab.bnc')

wordbank = Wordbank(vocab)
wordbank.load('wordbanks/wordbank.miro')
wordbank.load('wordbanks/wordbank.clean2')
wordbank.load('wordbanks/wordbank.2880')
print('done loading dictionary and wordbanks')
wordbank.save('output/wordbank.full')

# Read tokens and apply the first two wordbanks
if args.source_file is not None:
    with open(args.source_file) as fh:
        untokenized_ciphertext = fh.read()

        untokenized_ciphertext = apply_literals(untokenized_ciphertext, 'wordbanks/literals')
        ciphertext = tokenize_ciphertext(untokenized_ciphertext)
        ciphertext = [wordbank.apply(token) for token in ciphertext]

    if args.language_model == 'none':
        lattice = NoSubstitutionLattice(ciphertext)
        print('using no sub lattice')
    else:
        lattice = WilkinsonLattice(ciphertext, wordbank, args.beta, small_vocab)


elif args.lattice_file is not None:
    lattice = Lattice()
    lattice.from_carmel_lattice(args.lattice_file)

if args.language_model == 'gpt2':
    from gpt_lm import GPTLanguageModel # only import if necessary
    lm = GPTLanguageModel('gpt2', 'gpt2')
elif args.language_model == 'gpt2-large':
    from gpt_lm import GPTLanguageModel
    lm = GPTLanguageModel('gpt2-large', 'gpt2-large')
elif args.language_model == 'unigram':
    lm = UnigramLanguageModel()
elif args.language_model == 'length':
    lm = LengthLanguageModel()
elif args.language_model == 'ngram':
    from ngram_lm import NGramLanguageModel
    lm = NGramLanguageModel()
elif args.language_model == 'none' or args.language_model is None:
    if args.lattice_file is not None:
        raise ValueError('Must supply a language model if a lattice is provided')
    else:
        args.beam_width = 1
        lm = LengthLanguageModel()
else:
    # Provided a path to a fine-tuned GPT
    assert os.path.isdir(args.language_model)
    from gpt_lm import GPTLanguageModel
    lm = GPTLanguageModel(args.language_model, args.language_model)

if args.oracle:
    from oracle_lm import OracleLanguageModel
    args.beam_width = 1
    oracle = OracleLanguageModel(args.gold_file)
else:
    oracle = None

print('Loaded lattice and LM')

if args.language_model != 'none':
    lattice.to_carmel_lattice('output/lattices/unsolved.accuracy.lattice')
    print('saved lattice to file')

ScoreDrop = namedtuple("ScoreDrop", ['score_drop', 'plaintext', 'ciphertext'])
MAX_ITERATIONS = 10
STOP_PERCENTAGE = 90.0 # stop when this much of the wordbank is known

def score(message, gold):
    '''message, gold are both strings'''
    message_tokens = split_ciphertext(message)
    gold_tokens = split_ciphertext(gold)

    edit_distance = nltk.edit_distance(message_tokens, gold_tokens) # can just count tokens that mismatch, but use edit distance for robustness
    accuracy = 1 - edit_distance / len(gold_tokens)
    return accuracy

for step in range(MAX_ITERATIONS):
    # Count unk tokens. break if more than STOP_PERCENTAGE% of tokens are known
    unks = 0
    for token in ciphertext:
        if token.is_unk(wordbank):
            unks += 1
    print('unks', unks)
    if unks < len(ciphertext) * (1 - STOP_PERCENTAGE/100):
        break

    # Do beam search
    beam_result = beam_search(lm, lattice, beam_width=args.beam_width, oracle=oracle)

    # Confidence model:
    # Determines which words to add to the wordbank
    # Should return a list of ScoreDrop objects in sorted order
    drops = confidence_model(args, beam_result, wordbank)
    print('drops', drops)

    # add S substitutions to the wordbank
    for drop in drops:
        try:
            wordbank.add(drop.ciphertext, drop.plaintext, source='context')
            print('adding:', drop.ciphertext, drop.plaintext)
        except ValueError as e: # stop when tokens are out of order
            print('inconsistency', e)
            break

    # Dump the new wordbank
    wordbank.save('output/wordbank{}.full'.format(step))

    # Print edit distance at each step
    if args.gold_file is not None:

        with open(args.gold_file) as fh:
            gold_text = fh.read()

        print('Accuracy: ', score(lm.decode(beam_result[0].prediction), gold_text))

    if not args.self_learn:
        break
    else:
        # Set up new lattice
        if args.language_model == 'none':
            print('No substitution lattice shouldnt have self-learn enabled')
            break
        else:
            ciphertext = [wordbank.apply(token) for token in ciphertext]
            lattice = WilkinsonLattice(ciphertext, wordbank, args.beta, small_vocab)
            lattice.to_carmel_lattice('output/lattices/unsolved.accuracy{}.lattice'.format(step))
            print('saved lattice to file')


print('\n\n================ DONE ===============\n\n\n')
print('Final beams')
for beam in beam_result:
    print(str(beam))

print('Best decoding')
message = lm.decode(beam_result[0].prediction)
print(message)

if args.gold_file is not None:
    print('Final accuracy: ', score(message, gold_text))
