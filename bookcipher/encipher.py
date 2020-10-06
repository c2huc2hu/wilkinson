# Gathering results for the synthetic dataset
import argparse
from nltk.corpus import gutenberg
import spacy
nlp = spacy.load("en_core_web_sm") # need to do: python -m spacy download en_core_web_sm

parser = argparse.ArgumentParser(description='generate enciphered text')
parser.add_argument('wordbank_size', type=int, help='size of wordbank to use', default=800)

def encipher(plaintext, dictionary):
    '''
    Encipher a plaintext with a dictionary code. Words not in the Unix dictionary or proper nouns are 
    "Washington is my friend" -> [-1, 123, 456, 234]

    plaintext - a list of spacy tokens
    dictionary - a dict string -> index
    '''
    result = []

    for token in plaintext:

        lemmatized_token = token.lemma_.lower()
        if lemmatized_token == '-pron-':
            lemmatized_token = token.lower_ # don't lemmatize pronouns

        if lemmatized_token in dictionary:
            result.append(dictionary[lemmatized_token])
        elif token.lower_ in dictionary:
            result.append(dictionary[token.lower_])
        elif not (token.is_punct or token.is_space):
            # replace unk with -1
            result.append(-1)
        # else: pass # skip punctuation or spaces
    return result

def prep_ciphertext(ciphertext):
    '''
    Convert everything to table cipher, replacing unk with [100]^, preparing to dumy it to file

    When deciphering it, change MAX_WORD to your dictionary size in wordbank.py
    '''
    result = ' '.join('[{}]^'.format(i) if i != -1 else '[0]^' for i in ciphertext)
    return result

def make_dictionary(file):
    i = 0
    result = {}
    with open(file) as fh:
        for line in fh:
            if line.islower(): # unix dictionary contains a bunch of proper nouns. ignore them. note: this excludes "I" and proper nouns, but that's fine because it includes 'i' instead and the wilkinson dictionary also excludes proper nouns
                result[line.strip().lower()] = i
                i += 1
    return result

def make_wordbank(ciphertext, plaintext):
    '''
    ciphertext - a list of tokens (ints)
    plaintext - a list of spacy tokens
    '''

    assert len(ciphertext) == len(plaintext), 'ciphertext and plaintext must be the same length'

    result = {} # ciphertext id -> plaintext id
    for ct, pt in zip(ciphertext, plaintext):
        if ct != -1:
            if pt.lemma_ == '-PRON-':
                result[ct] = pt.lower_
            else:
                result[ct] = pt.lemma_.lower()
    return result

def dump_wordbank(wordbank, fh):
    '''dump wordbank to file'''
    for k, v in wordbank.items():
       fh.write('[{}]^\tsynthetic\t{}\n'.format(k, v))

def apply_wordbank(wordbank, ciphertext):
    '''dumb method to just apply wordbank'''
    result = []

    for ct in ciphertext:
        if ct != -1 and ct in wordbank:
            result.append(wordbank[ct])
        else:
            result.append(ct)
    return result

if __name__ == '__main__':

    args = parser.parse_args()
    N = args.wordbank_size # 800 is comparable to the actual cipher, 16998 is the full dataset

    dictionary = make_dictionary('/usr/share/dict/words')
    print('=== summary of dictionary === ')
    print(len(dictionary), 'words. put this in wordbank.py')

    # training_file = 'data/synthetic/wikipedia-washington.txt'
    # with open(training_file) as fh:
    #    pass

    training_file = 'chesterton-brown.txt'
    train_corpus = gutenberg.raw(training_file)

    print('== summary for training file {} =='.format(training_file))
    plaintext = [token for token in nlp(train_corpus) if not (token.is_punct or token.is_space)]
    ciphertext = encipher(plaintext, dictionary)

    plaintext = plaintext[:N]
    ciphertext = ciphertext[:N]
    wordbank = make_wordbank(ciphertext, plaintext)
    
    with open('data/synthetic/wordbank-{}'.format(N), 'w') as fh_out:
        dump_wordbank(wordbank, fh_out)

    print('number of tokens:', len(plaintext))
    print('number of types (excluding proper nouns):', len(wordbank))

    # test_file = 'data/synthetic/wikipedia-wilkinson.txt'
    # with open(test_file) as fh:
    test_file = 'chesterton-thursday.txt'
    test_corpus = gutenberg.raw(test_file)

    print('== summary for test file {} =='.format(test_file))
    test_plaintext = [token for token in nlp(test_corpus)[1000:2000] if not (token.is_punct or token.is_space)]
    with open('data/synthetic/test-plaintext.txt', 'w') as fh_out:
        fh_out.write(' '.join(token.lower_ for token in test_plaintext))
    test_ciphertext = encipher(test_plaintext, dictionary)
    print('number of tokens:', len(test_plaintext))
    test_wordbank = make_wordbank(test_ciphertext, test_plaintext)
    print('number of types', len(test_wordbank))

    # the number is the wordbank size
    with open('data/synthetic/test-ciphertext-{}.txt'.format(N), 'w') as fh_out:
        fh_out.write(prep_ciphertext(test_ciphertext))

    partially_deciphered_ct = apply_wordbank(wordbank, test_ciphertext)
    print('number of wordbanked tokens', sum([type(token) == str for token in partially_deciphered_ct]))
    print(' '.join(map(str, partially_deciphered_ct[:200])))
