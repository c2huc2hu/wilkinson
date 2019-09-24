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
        temperature = 0.7 # default: 0.7, increasing -> more novel results

        if not input_ids:
            input_ids = self.tokenizer.encode(self.tokenizer.bos_token) # can't feed empty arrays, so feed <BOS> instead

        with torch.no_grad(): # turn off backprop
            outputs = self.model(torch.tensor(input_ids).unsqueeze(0))  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states). TODO: benchmark
            next_token_logits = outputs[0][0, -1, :] / temperature
            output_probs = F.log_softmax(next_token_logits, dim=-1)
            return output_probs

class Beam():
    def __init__(self, prediction='', lm_prob=0, lattice_prob=0, lattice_idx=0):
        self.prediction = prediction
        self.lm_prob = lm_prob
        self.lattice_prob = lattice_prob
        self.log_prob = lm_prob + lattice_prob
        self.lattice_idx = 0 # position in lattice, since expand_beam iterates over wordpieces, need to keep track of this

    def map_prediction(self, fcn):
        return "{}: {} + {} = {}".format(fcn(self.prediction), self.lm_prob, self.lattice_prob, self.log_prob, self.lattice_idx)

    def __str__(self):
        return "{}: {} + {} = {}".format(self.prediction, self.lm_prob, self.lattice_prob, self.log_prob, self.lattice_idx)


def beam_search(source, lm, lattice, beam_width=8, alpha=1):
    '''
    Beam search over wordpieces

    Return a list of `beam_width` possible strings, sorted most to least probable
    '''
    lm.initialize(source)
    beams = [Beam(lm.tokenizer.encode(lm.tokenizer.bos_token))] # list of tokens
    exhausted_beams = []
    beam_idx = 0

    while beams:
        print('\n\nBeam iteration {}/{}'.format(beam_idx, len(source)))
        new_beams = []
        for beam in beams:
            # skip exhausted beams
            if beam.lattice_idx >= len(source):
                print('Exhausted source, ending')
                continue

            # if next token is a literal, add it to the beam directly
            # we still have to check all the suffixes for the previous token though
            if source[beam.lattice_idx].ciphertype is None:
                new_beams.append(Beam(beam.prediction + lm.tokenizer.encode(' ' + source[beam.lattice_idx].plaintext), beam.lm_prob, beam.lattice_prob, beam.lattice_idx + 1))
                print('literal. shouldnt be in test cases')
                quit()
                # todo: also add inflected forms here
            
            is_beam_finished = False
            lm_probs = lm.next_token(beam.prediction) # prior probability over all wordpieces

            _sentence_prefix, _space, last_word = lm.tokenizer.decode(beam.prediction).rpartition(' ')

            for (wordpiece, wordpiece_idx), lm_prob in zip(lm.tokenizer.encoder.items(), lm_probs):
                if wordpiece.startswith('Ġ'): # beginning of word symbol
                    advance_lattice = True # advance the lattice
                elif '\n' in wordpiece or wordpiece.isspace(): # skip spaces
                    print('continuing')
                    continue
                else:
                    advance_lattice = False # don't advance the lattice

                    # get the last word (fragment), narrow lattice probability according to tihs
                    last_word += wordpiece

                # apply lattice to the word so far and get the probability
                # the lattice is dynamically generated from wordpieces, and will not sum to 1
                # TODO: vectorize this
                source_token = source[beam.lattice_idx + advance_lattice]
                if not advance_lattice:
                    lattice_prob = lattice.integrate_prob(wordpiece[1:], source_token)
                else:
                    lattice_prob = lattice.integrate_prob(last_word + wordpiece, source_token) - lattice.integrate_prob(last_word, source_token)
                if lattice_prob == -np.inf:
                    continue # skip impossible cases

                # if a proposed beam proposes a new word past the end of our source_text, we're done with this beam
                # TODO: possible optimization. heapify to only keep top k. 
                #       can also use the lowest probability to prune some compute earlier
                if beam.lattice_idx + advance_lattice >= len(source):
                    if not is_beam_finished:
                        exhausted_beams.add(beam)
                        is_beam_finished = True

                # add new beam
                new_beam = Beam(beam.prediction + [wordpiece_idx], beam.lm_prob + alpha*lm_prob, beam.lattice_prob + lattice_prob, beam.lattice_idx + advance_lattice)
                new_beams.append(new_beam)

        print('Unpruned beams: (randomly select 10/{})'.format(len(new_beams)))
        import random
        for b in random.choices(new_beams, k=min(10, len(new_beams))):
            print(repr(b.map_prediction(lm.tokenizer.decode)))

        beams = sorted(new_beams, key=lambda b: -b.log_prob)[:beam_width]
        print('Beams:\n')
        for i in range(min(10, len(beams))):
            print('Beam # ', i)
            print(beams[i].prediction, repr(beams[i].map_prediction(lm.tokenizer.decode)))
        beam_idx += 1

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
    print('Best predictions (in order)', beam_search(ct, lm, lattice, alpha=1, beam_width=4))