# Taken from our spell corrector

import argparse
import math
import os
import json
from .dataset import get_source_target_lines

class Lattice():
    '''Represents substitution probabilites in a string given the source. Insertions and deletions are not supported'''

    def __init__(self, train_files=None, topN=None, lattice_file=None):
        if train_files is not None:
            self.learn_alignments(train_files, topN)
            if lattice_file is not None:
                with open(lattice_file, 'w') as f:
                    json.dump(self.alignments, f)
        elif lattice_file is not None:
            with open(lattice_file, 'r') as f:
                self.alignments = json.load(f)

    def learn_alignments(self, train_files, topN):
        self.alignments = {}
        correction_counts = {}
        for i, (source_sentence, target_sentence) in enumerate(get_source_target_lines(train_files)):
            if not i % 10000: print(i)
            for (source_char, target_char) in zip(source_sentence, target_sentence):
                if not source_char in self.alignments:
                    self.alignments[source_char] = {}
                if not target_char in self.alignments[source_char]:
                    self.alignments[source_char][target_char] = 0
                self.alignments[source_char][target_char] += 1
                if not target_char in correction_counts:
                    correction_counts[target_char] = 0
                correction_counts[target_char] += 1
        for source_char in self.alignments:
            top_targets = sorted([(target, count) for (target, count) in self.alignments[source_char].items()], key=lambda k: -k[1])
            if topN is not None:
                top_targets = top_targets[:topN]
            self.alignments[source_char] = {t[0]: math.log(t[1]/correction_counts[t[0]]) for t in top_targets}

    def backward_probs(self, mistake_char, correction_char):
        '''Log probability of correction_char given mistake_char, or decoding given the actual state'''
        if mistake_char in self.alignments:
            return self.alignments[mistake_char].get(correction_char, -1000)
        return -1000

    def possible_substitutions(self, mistake_char):
        '''Return a list of possible candidates given an observed state. Basically a mask for the language model'''
        if mistake_char in self.alignments:
            return tuple(self.alignments[mistake_char].keys())
        else:
            return (mistake_char,)

    def write_lattice_files(self, test_file, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        with open(test_file, "r") as test:
            for i, line in enumerate(test):
                if not i % 1000: print(i)
                line = line.rstrip('\n')
                with open(os.path.join(output_folder, "test_{}.wlat".format(i)), "w") as f:
                    f.write("name ex_{}\n".format(i))
                    f.write("numaligns {}\n".format(len(line)))
                    f.write("posterior 1\n")
                    for char_index, char in enumerate(line):
                        if not char in self.alignments: self.alignments[char] = [(char, 1.0)]
                        targets = self.alignments[char]
                        f.write("align {}".format(char_index))
                        for target in targets:
                            f.write(" {} {}".format(target, targets[target]))
                        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make lattice of common corrections")
    parser.add_argument("--input_files", type=str, nargs='+')
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--output_folder", type=str)
    args = parser.parse_args()

    lattice = Lattice(args.input_files)
