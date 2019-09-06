from wordbank import Wordbank, Token
from vocab import Vocab

from decipher import tokenize_ciphertext # todo: move this

# One hour baseline

wordbank = Wordbank()
wordbank.load('wordbanks/wordbank.clean')
wordbank.load('wordbanks/wordbank.guess')
vocab = Vocab('dict.modern')

# ciphertext = '[556]^ 586.[26]- Ferdinand 2 [678]^ 95 ? [1235]^ y ? [433]^ [79]^ [664]^'

with open('data/unsolved.ciphers.1078') as fh:
    ciphertext = fh.read()

# raw text
tokenized_ct = tokenize_ciphertext(ciphertext)
# print('raw text')
# print(tokenized_ct)
# print('================== raw text')

# apply wordbank
wordbanked_ct = [wordbank.apply(token) for token in tokenized_ct]
# print('wordbanked text')
# print(wordbanked_ct)
# print('================== wordbanked text')

# apply a basic interpolation
deciphered_ct = [
    wordbank.interpolate(token, vocab) if token.is_unk() else token for token in wordbanked_ct
]

# print('deciphered text')
# print(deciphered_ct)
# print('================== deciphered text')

# print(' '.join(map(str, deciphered_ct)))


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

    fh.write(' '.join(token.to_html() for token in deciphered_ct))

    fh.write('</body>')