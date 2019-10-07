from wordbank import Wordbank, Token
from vocab import Vocab
from passes import tokenize_ciphertext, add_frequency_attack, beam_search_pass, dump_lattice

from word_beamsearch import token_beam_search, GPTLanguageModel, TokenLattice

BEAM_WIDTH = 2

print(f'Config: {BEAM_WIDTH}')

# Runtime
with open('data/unsolved.ciphers.accuracy') as fh:
    untokenized_ciphertext = fh.read()
    ciphertext = tokenize_ciphertext(untokenized_ciphertext)

# TESTING THINGS
# ciphertext = tokenize_ciphertext('501.[20]= [804]^ \n [1218]^ 140.[8]- [426]^')

vocab = Vocab('dict.modern')
wordbank = Wordbank(vocab)
wordbank.load('wordbanks/wordbank.miro')
wordbank.load('wordbanks/wordbank.clean')
wordbank.load('wordbanks/wordbank.2880')
wordbank.load('wordbanks/wordbank.guess')
print('done loading dictionary and wordbanks')

# apply the first two wordbanks
ciphertext = [wordbank.apply(token) for token in ciphertext]
print(ciphertext)

lm = GPTLanguageModel(vocab)
lattice = TokenLattice(wordbank)
beam_result = token_beam_search(ciphertext, lm, lattice, beam_width=BEAM_WIDTH)

print('\n\n================ DONE ===============\n\n\n')
for beam in beam_result:
    print(str(beam))

message = lm.tokenizer.decode(beam[0])

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
