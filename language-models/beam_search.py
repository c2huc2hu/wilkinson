# Taken from our spell corrector

import argparse
from lattices import Lattice
import math

class Beam():
    def __init__(self, prediction='', lm_prob=0, lattice_prob=0):
        self.prediction = prediction
        self.lm_prob = lm_prob
        self.lattice_prob = lattice_prob
        self.log_prob = lm_prob + lattice_prob

    def __str__(self):
        return "{}: {} + {} = {}".format(self.prediction, self.lm_prob, self.lattice_prob, self.log_prob)

class LanguageModel():
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_indices = {self.vocab[i]: i for i in range(len(self.vocab))}

    def initialize(self, source):
        self.source = source

    def next_token(self, s):
        pass

def beam_search(source, lm, lattice, beam_width=8, alpha=1):
    lm.initialize(source)
    beams = [Beam()]
    for i in range(len(source)):
        # for beam in beams: print(beam)
        new_beams = []
        for beam in beams:
            new_beams += _expand_beam(beam, lm, lattice, source[i], alpha)
        beams = sorted(new_beams, key=lambda b: -b.log_prob)[:beam_width]
    for beam in beams: print(beam)
    return [beam.prediction for beam in beams]

def _expand_beam(beam, lm, lattice, mistake_char, alpha):
    next_beams = []
    word_probs = lm.next_token(beam.prediction)
    for next_char in lattice.possible_substitutions(mistake_char):
        lattice_prob = lattice.backward_probs(mistake_char, next_char)
        lm_prob = float(word_probs[lm.vocab_indices.get(next_char, 0)])
        new_beam = Beam(beam.prediction + next_char, beam.lm_prob + alpha*lm_prob, beam.lattice_prob + lattice_prob)
        next_beams.append(new_beam)
    return next_beams

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make lattice of common corrections")
    parser.add_argument("--train_files", type=str, nargs='+')
    parser.add_argument("--topN", type=int, default=None)
    parser.add_argument("--lattice_file", type=str)
    args = parser.parse_args()

    class FoxLanguageModel(LanguageModel):
        def __init__(self):
            super().__init__(('f', 'o', 'x'))

        def next_token(self, s):
            if len(s) == 0:
                return [-0.1, -10, -10]
            elif s[-1] == 'f':
                return [-10, -0.1, -0.1]
            elif s[-1] == 'o':
                return [-10, -10, -0.1]
            elif s[-1] == 'x':
                return [-10, -10, -10]

    class FoxLattice(Lattice):
        def __init__(self):
            self.lattice_probs = {
                ('f', 'f'): -0.1,
                ('f', 'o'): -1,
                ('f', 'x'): -1,
                ('o', 'f'): -1,
                ('o', 'o'): -0.1,
                ('o', 'x'): -1,
                ('x', 'f'): -1,
                ('x', 'o'): -1,
                ('x', 'x'): -0.1
            }

        def backward_probs(self, source_char, target_char):
            return self.lattice_probs.get((source_char, target_char), -1000)

        def possible_substitutions(self, mistake_char):
            return tuple('fox')

    lm = FoxLanguageModel()
    lattice = FoxLattice()
    print('Best predictions (in order)', beam_search('fxo', lm, lattice, alpha=1))
