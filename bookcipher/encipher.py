# Gathering results for the synthetic dataset

import spacy
nlp = spacy.load("en_core_web_sm") # need to do: python -m spacy download en_core_web_sm

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
        if lemmatized_token in dictionary:
            result.append(dictionary[lemmatized_token])
        elif token.lower_ in dictionary:
            result.append(dictionary[token.lower_])
        elif not (token.is_punct or token.is_space):
            # replace unk with -1
            result.append(-1)
        # else: pass # skip punctuation or spaces
    return result

def make_dictionary(file):
    i = 0
    result = {}
    with open(file) as fh:
        for line in fh:
            if not line.istitle(): # unix dictionary contains a bunch of proper nouns. ignore them.
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
            result[ct] = pt.lemma_.lower()
    return result

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
    # dictionary = make_dictionary('data/synthetic/dictionary-toy.txt')
    dictionary = make_dictionary('/usr/share/dict/words')
    N = 800 # 800 is comparable to the actual cipher, 16998 is the full dataset

    training_file = 'data/synthetic/wikipedia-washington.txt'
    with open(training_file) as fh:
        print('== summary for training file {} =='.format(training_file))
        plaintext = [token for token in nlp(fh.read()) if not (token.is_punct or token.is_space)]
        ciphertext = encipher(plaintext, dictionary)

        plaintext = plaintext[:N]
        ciphertext = ciphertext[:N]
        wordbank = make_wordbank(ciphertext, plaintext)

        print('number of tokens:', len(plaintext))
        print('number of types (excluding proper nouns):', len(wordbank))

    test_file = 'data/synthetic/wikipedia-wilkinson.txt'
    with open(test_file) as fh:
        print('== summary for test file {} =='.format(test_file))
        test_plaintext = [token for token in nlp(fh.read()) if not (token.is_punct or token.is_space)]
        test_ciphertext = encipher(test_plaintext, dictionary)
        print('number of tokens:', len(test_plaintext))
        test_wordbank = make_wordbank(test_ciphertext, test_plaintext)
        print('number of types', len(test_wordbank))

        partially_deciphered_ct = apply_wordbank(wordbank, test_ciphertext)
        print('number of wordbanked tokens', sum([type(token) == str for token in partially_deciphered_ct]))
        print(' '.join(map(str, partially_deciphered_ct[:200])))