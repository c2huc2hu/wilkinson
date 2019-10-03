import heapq

from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F
import numpy as np
from language_models.beam_search import LanguageModel


class GPTLanguageModel(LanguageModel):
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        self.model.eval() # set model to eval mode because we're not fine tuning

        self.vocab = self.tokenizer.encode
        self.unk = self.tokenizer.unk_token

        self.score_calls = 0


    def next_token(self, input_ids):
        '''
        Get the log probabilities of the next wordpiece
        s should be an iterable of ints, the result of calling self.tokenizer.encode()
        '''
        temperature = 1 # default: 0.7, increasing -> more novel results

        if not input_ids:
            input_ids = self.tokenizer.encode(self.tokenizer.bos_token) # can't feed empty arrays, so feed <BOS> instead

        with torch.no_grad(): # turn off backprop
            outputs = self.model(torch.tensor(input_ids).unsqueeze(0))  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states). TODO: benchmark
            next_token_logits = outputs[0][0, -1, :] / temperature
            output_probs = F.log_softmax(next_token_logits, dim=-1)
            return output_probs

# class Beam():
#     def __init__(self, prediction='', lm_prob=0, lattice_prob=0, lattice_idx=0):
#         self.prediction = prediction
#         self.lm_prob = lm_prob
#         self.lattice_prob = lattice_prob
#         self.log_prob = lm_prob + lattice_prob
#         self.lattice_idx = 0 # position in lattice, since expand_beam iterates over wordpieces, need to keep track of this

#     def map_prediction(self, fcn):
#         return "{}: {} + {} = {}".format(fcn(self.prediction), self.lm_prob, self.lattice_prob, self.log_prob, self.lattice_idx)

#     def __str__(self):
#         return "{}: {} + {} = {}".format(self.prediction, self.lm_prob, self.lattice_prob, self.log_prob, self.lattice_idx)


# def beam_search(source, lm, lattice, beam_width=8, alpha=1):
#     '''
#     Beam search over wordpieces

#     Return a list of `beam_width` possible strings, sorted most to least probable
#     '''
#     lm.initialize(source)
#     beams = [Beam(lm.tokenizer.encode(lm.tokenizer.bos_token))] # list of tokens
#     exhausted_beams = []
#     beam_idx = 0

#     while beams:
#         print('\n\nBeam iteration {}/{}'.format(beam_idx, len(source)))
#         new_beams = []
#         for beam in beams:
#             # skip exhausted beams
#             if beam.lattice_idx >= len(source):
#                 print('Exhausted source, ending')
#                 continue

#             # if next token is a literal, add it to the beam directly
#             # we still have to check all the suffixes for the previous token though
#             if source[beam.lattice_idx].ciphertype is None:
#                 new_beams.append(Beam(beam.prediction + lm.tokenizer.encode(' ' + source[beam.lattice_idx].plaintext), beam.lm_prob, beam.lattice_prob, beam.lattice_idx + 1))
#                 print('literal. shouldnt be in test cases')
#                 quit()
#                 # todo: also add inflected forms here
            
#             is_beam_finished = False
#             lm_probs = lm.next_token(beam.prediction) # prior probability over all wordpieces

#             _sentence_prefix, _space, last_word = lm.tokenizer.decode(beam.prediction).rpartition(' ')

#             for (wordpiece, wordpiece_idx), lm_prob, i in zip(lm.tokenizer.encoder.items(), lm_probs, range(100)):
#                 print(f'Considering wordpiece: {wordpiece} #{wordpiece_idx}')

#                 # import pdb
#                 # pdb.set_trace()

#                 if wordpiece.startswith('Ġ'): # beginning of word symbol
#                     advance_lattice = True # advance the lattice
#                     print('advancing lattice')
#                 elif any(ch in wordpiece for ch in ('\\\n\t@#$%^&*()`~[]{}-_=+')) or wordpiece.isspace(): # skip blank characters and punctuation
#                     print('continuing')
#                     continue
#                 else:
#                     advance_lattice = False # don't advance the lattice
#                     print('not advancing lattice')

#                     # narrow probability according to this
                    

#                 # apply lattice to the word so far and get the probability
#                 # the lattice is dynamically generated from wordpieces, and will not sum to 1
#                 # TODO: vectorize this
#                 source_token = source[beam.lattice_idx + advance_lattice]
#                 print('token', repr(source_token))

#                 if not advance_lattice:
#                     new_prob = lattice.integrate_prob(wordpiece[1:], source_token)
#                     old_prob = 0
#                 else:
#                     new_prob = lattice.integrate_prob(last_word + wordpiece, source_token)
#                     old_prob = lattice.integrate_prob(last_word, source_token)
#                 if new_prob == -np.inf:
#                     continue # skip impossible cases
#                 else:
#                     lattice_prob = old_prob - new_prob

#                 print('lattice prob')

#                 # if a proposed beam proposes a new word past the end of our source_text, we're done with this beam
#                 # TODO: possible optimization. heapify to only keep top k. 
#                 #       can also use the lowest probability to prune some compute earlier
#                 if beam.lattice_idx + advance_lattice >= len(source):
#                     print('shouldnt get here')
#                     if not is_beam_finished:
#                         exhausted_beams.add(beam)
#                         is_beam_finished = True

#                 # add new beam
#                 new_beam = Beam(beam.prediction + [wordpiece_idx], beam.lm_prob + alpha*lm_prob, beam.lattice_prob + lattice_prob, beam.lattice_idx + advance_lattice)
#                 new_beams.append(new_beam)
#                 print('added new beam', new_beam)
    
#         print('Unpruned beams: (randomly select 10/{})'.format(len(new_beams)))
#         import random
#         for b in random.choices(new_beams, k=min(10, len(new_beams))):
#             print(repr(b.map_prediction(lm.tokenizer.decode)))

#         beams = sorted(new_beams, key=lambda b: -b.log_prob)[:beam_width]
#         print('Beams:\n')
#         for i in range(min(10, len(beams))):
#             print('Beam # ', i)
#             print(beams[i].prediction, repr(beams[i].map_prediction(lm.tokenizer.decode)))
#         beam_idx += 1

#     return [lm.tokenizer.decode(beam.prediction) for beam in beams]

class FixedSizeHeap(list):
    def __init__(self, max_size):
        super().__init__([]) # min heap. make sure to maintain the heap invariant
        self.max_size = max_size

    def append(self, elem):
        '''Add an element and 
        heappushpop()
        '''
        if len(self) < self.max_size:
            heapq.heappush(self, elem)
        else:
            heapq.heappushpop(self, elem)

    def __min___(self):
        return self[0]


class GPTBeam():
    tokenizer = None # GPT tokenizer

    def __init__(self, prediction=[], lm_prob=0, lattice_prob=0, last_word_prob=0, lattice_idx=0, exhausted=False):
        self.prediction = prediction
        self.lm_prob = lm_prob
        self.lattice_prob = lattice_prob
        self.log_prob = lm_prob + lattice_prob
        self.lattice_idx = lattice_idx # position in lattice, since expand_beam iterates over wordpieces, need to keep track of this

        self.last_word_prob = last_word_prob # lattice probability of the last word. needs to be stored for GPT
        self.exhausted = exhausted

    def __str__(self):
        if GPTBeam.tokenizer is None:
            return "{}:\n    {} + {} = {} (exhausted: {}, lattice_idx {})".format(self.prediction, self.lm_prob, self.lattice_prob, self.log_prob, self.lattice_idx, self.exhausted, self.lattice_idx)
        else:
            return "{}/{}:\n    {} + {} = {} (exhausted: {}, lattice_idx {})".format(self.prediction, GPTBeam.tokenizer.decode(self.prediction), self.lm_prob, self.lattice_prob, self.log_prob, self.exhausted, self.lattice_idx)

    def __lt__(self, other):
        return self.log_prob < other.log_prob


def beam_search(source, lm, lattice, beam_width=8, alpha=1):
    '''
    Beam search over wordpieces

    Return a list of `beam_width` possible strings, sorted most to least probable
    '''

    lm.initialize(source)
    beams = [GPTBeam(lm.tokenizer.encode('I wrote you on the 23 November and warned you of a meditated attack against Louisiana by the people of Kentucky at the instigation of the late minister of the French at the Court of the United States: '))] # seed with something taken out of a random letter
    GPTBeam.tokenizer = lm.tokenizer

    # Separate BERT vocab into suffixes and words
    suffixes = {} # wordpieces that can't start a word. mapping suffix -> index in BERT vocab
    words = {} # wordpieces that can start a word. mapping word -> index in BERT vocab
    beam_idx = 0
    for wordpiece, wordpiece_idx in lm.tokenizer.encoder.items():

        # drop wordpieces except mandatory punctation in English (comma, space, apostrophe)
        if any(ch in wordpiece for ch in ('!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\tĊ 1234567890')):
            continue
        elif wordpiece == 'Ġ' or 'Ċ' in wordpiece:
            # whitespace characters, Ċ is newline, Ġ is lone space
            continue
        elif wordpiece.startswith('Ġ'):
            words[wordpiece[1:]] = wordpiece_idx # drop Ġ from the string representation
        else:
            suffixes[wordpiece] = wordpiece_idx


    while any(not beam.exhausted for beam in beams):
         # This will take a while. Print progress
        beam_idx += 1

        # TEMP
        if beam_idx >= 100:
            break
        # END TEMP

        print('\n\nBeam iteration {}/{}'.format(beam_idx, len(source)))

        new_beams = FixedSizeHeap(beam_width)
        for beam in beams:
            # If beam is exhausted, we're done with no changes
            if beam.exhausted:
                new_beams.append(beams)
                continue

            # At each step of the beam, we consider two cases:
            # 1) add a new word and advance the source_token
            # 2) add an affix and don't advance the source_token

            # Precompute GPT-2 probabilites
            lm_probs = lm.next_token(beam.prediction)

            # Case 1: Advance the source token
            lattice_idx = beam.lattice_idx + 1

            # Skip beams where we've exhausted all tokens
            if lattice_idx >= len(source):
                print('Exhausted source, ending')
                beam.exhausted = True
                new_beams.append(beam)

            # If the token is a literal, add it and inflected forms to the beam directly
            elif source[lattice_idx].ciphertype is None:
                new_beam = GPTBeam(beam.prediction + lm.tokenizer.encode(source[lattice_idx].plaintext), beam.lm_prob, beam.lattice_prob + 0, 0, lattice_idx)
                # print('Ciphertype is non')
                # raise Exception('shouldnt get here')
                # TODO: also add infected forms
                # TODO: correctly update lm_prob so it can be compared accurately

            # Otherwise consider new words that can extend the beam
            else:
                lattice_probs = lattice.integrate_probs_batch(words, source[lattice_idx])

                for i, (word, word_idx) in enumerate(words.items()):
                    lm_prob = lm_probs[word_idx]
                    lattice_prob = lattice_probs[i] - beam.last_word_prob

                    if lattice_prob > -np.inf:
                        new_beams.append(GPTBeam(beam.prediction + [word_idx], beam.lm_prob + alpha*lm_prob, beam.lattice_prob + lattice_prob, lattice_prob, lattice_idx))

            # Case 2: Add an affix. This is always a possibility except after the first word
            if beam_idx > 1:

                lattice_idx = beam.lattice_idx
                _sentence, _space, last_word = lm.tokenizer.decode(beam.prediction).rpartition(' ')

                # Add all affixes that are still compatible with the source token
                lattice_probs = lattice.integrate_probs_batch((last_word + suffix for suffix in suffixes), source[lattice_idx])
                for i, (suffix, suffix_idx) in enumerate(suffixes.items()):
                    lm_prob = lm_probs[suffix_idx]
                    lattice_prob = lattice_probs[i]

                    if lattice_prob > -np.inf:
                        new_beams.append(GPTBeam(beam.prediction + [suffix_idx], beam.lm_prob + alpha*lm_prob, beam.lattice_prob + lattice_prob - beam.last_word_prob, beam.last_word_prob + lattice_prob, lattice_idx))

        # Print a sample of beams for debugging
        print('Unpruned beams: (randomly select 10/{})'.format(len(new_beams)))
        import random
        for b in random.choices(new_beams, k=min(10, len(new_beams))):
            print(b)

        # Keep the best beam_width beams, including exhausted beams. TODO: heapify this
        beams = sorted(new_beams, reverse=True)[:beam_width]

        # Print the best beams
        print('Beams (best to worst):\n')
        for i in range(min(beam_width, 10, len(beams))):
            print('Beam # ', i)
            print(beams[i])

    return [lm.tokenizer.decode(beam.prediction) for beam in beams]

if __name__ == '__main__':
    from passes import tokenize_ciphertext
    from vocab import Vocab
    from wordbank import Wordbank, Token
    from token_lattice import TokenLattice

    v = Vocab('wordbanks/vocab.toy')
    wb = Wordbank(v)
    wb.load('wordbanks/wordbank.toy')
    ct = tokenize_ciphertext('[330]^ [960]^ [1078]^ [490]^') # given the toy vocab, should give "a cat sees the egg"

    lm = GPTLanguageModel()
    lattice = TokenLattice(wb)

    # some basic tests
    assert (-np.inf
        == lattice.integrate_prob('longing', Token('[1000]^')) # impossible guess
        == lattice.integrate_prob('longing', Token('[728]^')) # impossible guess
        < lattice.integrate_prob('longing', Token('[990]^')) # terrible guess
        < lattice.integrate_prob('longing', Token('[731]^')) # bad guess
        < lattice.integrate_prob('longing', Token('[733]^')) # good guess
        < 0)

    ct = [wb.apply(tok) for tok in ct]

    # print(ct)
    print('Best predictions (in order)', beam_search(ct, lm, lattice, alpha=1, beam_width=16))