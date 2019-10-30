from wordbank import Wordbank, Token
from vocab import Vocab
from passes import apply_literals, tokenize_ciphertext, add_frequency_attack, beam_search_pass, dump_lattice

from general_beamsearch import beam_search, GPTLanguageModel, LengthLanguageModel, UnigramLanguageModel, Lattice
from wilkinson_lattice import WilkinsonLattice, NoSubstitutionLattice

# Load config
import argparse
parser = argparse.ArgumentParser(description='Solve a bookcipher')
parser.add_argument('-b', '--beam-width', nargs='?', default=2, help='width of beam search. runtime scales linearly', type=int)
parser.add_argument('--lattice_file', nargs='?', help='path to lattice file')
parser.add_argument('--source_file', metavar='source-file', help='source file to decode')
parser.add_argument('--gold_file', metavar='gold-file', help='reference translation for scoring accuracy')
parser.add_argument('--language-model', '--lm', help='which language model to use', choices=['gpt2', 'gpt2-large', 'unigram', 'length', 'none'])
args = parser.parse_args()

if args.source_file is None and args.lattice_file is None:
    args.source_file = 'data/unsolved.ciphers.accuracy.clean'
    args.gold_file = 'data/unsolved.ciphers.accuracy.gold'
if args.source_file is not None and args.lattice_file is not None:
    raise ValueError('Cannot supply both a lattice and source file')
print('Config:', args)

# Runtime
vocab = Vocab('dict.modern')

wordbank = Wordbank(vocab)
wordbank.load('wordbanks/wordbank.miro')
wordbank.load('wordbanks/wordbank.clean')
wordbank.load('wordbanks/wordbank.2880')
wordbank.load('wordbanks/wordbank.guess')
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
        lattice = WilkinsonLattice(ciphertext, wordbank)


elif args.lattice_file is not None:
    lattice = Lattice()
    lattice.from_carmel_lattice(args.lattice_file)

# print(ciphertext)
# for tok in ciphertext:
#     print(tok, tok.ciphertype)


if args.language_model == 'gpt2':
    lm = GPTLanguageModel('/nfs/cold_project/users/chrischu/data/pytorch-transformers/gpt2', '/nfs/cold_project/users/chrischu/data/pytorch-transformers/gpt2')
elif args.language_model == 'gpt2-large':
    lm = GPTLanguageModel('/nfs/cold_project/users/chrischu/data/pytorch-transformers/gpt2-large', '/nfs/cold_project/users/chrischu/data/pytorch-transformers/gpt2-large')
elif args.language_model == 'unigram':
    lm = UnigramLanguageModel()
elif args.language_model == 'length':
    lm = LengthLanguageModel()
elif args.language_model == 'none':
    if args.lattice_file is not None:
        raise ValueError('Must supply a language model if a lattice is provided')
    else:
        args.beam_width = 1
        lm = LengthLanguageModel()
else:
    raise ValueError('Invalid language model', args.language_model)

print('Loaded lattice and LM')

if args.language_model != 'none':
    lattice.to_carmel_lattice('output/lattices/unsolved.accuracy.lattice')
    print('saved lattice to file')

beam_result = beam_search(lm, lattice, beam_width=args.beam_width)

print('\n\n================ DONE ===============\n\n\n')
for beam in beam_result:
    print(str(beam))

message = lm.decode(beam_result[0].prediction)

if args.gold_file is not None:
    with open(args.gold_file) as fh:
        gold_text = fh.read()
        gold_tokens = gold_text.split()

    message_tokens = message.split()
    import nltk
    accuracy = nltk.edit_distance(message_tokens, gold_tokens) # can just count tokens that mismatch, but use edit distance for robustness
    print('Edit distance', accuracy)

print('done')
quit(0)

# super hacky html output
import os
os.makedirs('output', exist_ok=True)
with open('output/visualize.html', 'w') as fh:
    fh.write('''
        <head>
            <link rel="stylesheet" type="text/css" href="../display/style.css" />
        </head>

        <body>
        <p>
            <div class="key" style="background-color:lightblue">from word bank</div>
            <div class="key" style="background-color:turquoise">literal</div>
            <div class="key" style="background-color:rgb(0,127,0)">uncertain guess (p=0)</div>
            <div class="key" style="background-color:rgb(0,255,0)">certain guess (p=1)</div>

        </p>

        ''')

    fh.write(' '.join(token.to_html() for token in message))

    fh.write('</body>')

print('done!')
