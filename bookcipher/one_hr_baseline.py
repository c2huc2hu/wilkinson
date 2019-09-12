from wordbank import Wordbank, Token
from vocab import Vocab
from passes import tokenize_ciphertext, add_frequency_attack

# One hour baseline

## Debugging string
# ciphertext = '[556]^ 586.[26]- Ferdinand 2 [678]^ 95 ? [1235]^ y ? [433]^ [79]^ [664]^'

with open('data/unsolved.ciphers.1078') as fh:
    untokenized_ciphertext = fh.read()
    ciphertext = tokenize_ciphertext(untokenized_ciphertext)


wordbank = Wordbank()
wordbank.load('wordbanks/wordbank.miro')
wordbank.load('wordbanks/wordbank.clean')
wordbank.load('wordbanks/wordbank.guess')
vocab = Vocab('dict.modern')

print(wordbank._dict)

# apply the first two wordbanks
ciphertext = wordbank.run(ciphertext)

# apply a frequency attack
# add_frequency_attack(wordbank, ciphertext)

# interpolate to make it readable
message = wordbank.run(ciphertext, interpolate=True, vocab=vocab)



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