#!/usr/bin/env python3
from __future__ import division
import re
import nltk
import sys


split_re = re.compile(r'''
    \s
    |(\d+\.\[\d+\][=-]) # dict cipher
    |(\[\d+\]\^)  # table cipher
''', flags=re.VERBOSE)
def split_ciphertext(ciphertext):
    return [match for match in re.split(split_re, ciphertext) if match]

def score(message, gold):
    '''message, gold are both strings'''
    # gpt decodes punctuation attached to text: e.g. "hello, world.", but the oracle would decode it: "hello , world ."
    # the score doesn't change because the location of periods is always known
    message = message.replace(' .', '.')
    gold = gold.replace(' .', '.')
    message_tokens = split_ciphertext(message)
    gold_tokens = split_ciphertext(gold)
    edit_distance = nltk.edit_distance(message_tokens, gold_tokens) # can just count tokens that mismatch, but use edit distance for robustness
    accuracy = 1 - edit_distance / len(gold_tokens)
    return accuracy

if __name__ == '__main__':
    if len(sys.argv) > 1:
        gold_path = sys.argv[1] # path to the gold text
    else:
        gold_path = 'data/eval/plaintext.txt'

    with open(gold_path) as fh:
        gold = re.sub('\s+', ' ', fh.read()).strip()

    for line in sys.stdin:
        sys.stdout.write(str(score(line.strip(), gold)) + '\n')
