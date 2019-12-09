
# Confidence models
# Determine which words to add to the wordbank for self-learning

from collections import namedtuple
from passes import split_ciphertext

ScoreDrop = namedtuple("ScoreDrop", ['score_drop', 'plaintext', 'ciphertext'])

def confidence_model(args, beam_result, wordbank):
    '''
    Look for unknown tokens that give the smallest drop in language model + lattice score,
    and add those to the wordbank.

    This only uses the left context.

    Returns a list of ScoreDrop objects to add to the wordbank
    '''

    # Get training history
    best_history = []
    best_beam = beam_result[0]
    while best_beam is not None:
        best_history.append(best_beam)
        best_beam = best_beam.prev
    best_history.reverse()

    # Get score drops
    prev_prob = 0
    prev_len = 0 # length of deciphered sequence in characters
    drops = []

    for i, beam in enumerate(best_history[1:]): # first element contains empty beam

        # wilkinson specific properties
        token = beam.lattice_edge.token
        raw_form = beam.lattice_edge.uninflected_form
        new_word = beam.lattice_edge.label.lower()

        score_drop = prev_prob - beam.log_prob # positive
        prev_prob = beam.log_prob

        if args.confidence_model == 'left':
            if token.is_unk(wordbank): # only add unknown tokens to wordbank 
                drops.append(ScoreDrop(score_drop=score_drop, plaintext=new_word, ciphertext=token.raw))

        elif args.confidence_model == 'oracle':
            # Oracle confidence model:
            # Add all correct words to the wordbank. This provides an upper bound given our language model

            with open(args.gold_file) as fh:
                content = fh.read()
                gold_tokens = split_ciphertext(content) #  + ['a'] * 1

            if token.is_unk(wordbank) and gold_tokens[i] == new_word:
                drops.append(ScoreDrop(score_drop=0, plaintext=new_word, ciphertext=token.raw))

    drops.sort(key=lambda x: x.score_drop)

    # add S substitutions to the wordbank
    return drops[:args.substitutions]