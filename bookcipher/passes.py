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


# bigram attack
# get every word tha appears in a 