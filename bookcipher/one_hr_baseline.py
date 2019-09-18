from wordbank import Wordbank, Token
from vocab import Vocab
from passes import tokenize_ciphertext, add_frequency_attack, beam_search_pass

# One hour baseline

## Debugging string
# ciphertext = '[556]^ 586.[26]- Ferdinand 2 [678]^ 95 ? [1235]^ y ? [433]^ [79]^ [664]^'

with open('data/unsolved.ciphers.1078') as fh:
    untokenized_ciphertext = fh.read()
    ciphertext = tokenize_ciphertext(untokenized_ciphertext)

# TEMP
# ciphertext = tokenize_ciphertext('501.[20]= [804]^ \n [1218]^ 140.[8]- [426]^')

vocab = Vocab('dict.modern')
wordbank = Wordbank(vocab)
wordbank.load('wordbanks/wordbank.miro')
wordbank.load('wordbanks/wordbank.clean')
wordbank.load('wordbanks/wordbank.2880')
wordbank.load('wordbanks/wordbank.guess')

# apply the first two wordbanks
ciphertext = wordbank.run(ciphertext)

# apply a frequency attack
# add_frequency_attack(wordbank, ciphertext)

# do a beam search with a language model
decoded_text = beam_search_pass(ciphertext, wordbank, beam_width=1)

# interpolate to make it readable
# message = wordbank.run(ciphertext, interpolate=True)


# join tokens to decoded_text
for plaintext, token in zip(decoded_text, ciphertext):
    token.plaintext = plaintext

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