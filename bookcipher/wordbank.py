# functions for handling wordbanks

def load_wordbank(filename, source_filter=None):
    '''
    Load a wordbank file into a dictionary {number: }

    filename: path to wordbank file. data should be tab separated in 3 columns: (location, source, word)
    source_filter: list of sources to include. useful if multiple ciphers were used. defaults to including everything
    '''
    # TODO: Future improvement: one more letters are wordbanked, it would be useful to automatically
    # partition letters according to which dictionary they came from. should be pretty straightforward

    wordbank = {}

    with open(filename) as f:
        next(f) # skip header line

        for line in f:
            try:
                location, source, word = line.strip().split('\t')
            except ValueError:
                pass # blank line; skip

            if source_filter is None or source in source_filter:
                wordbank[location] = word

    return wordbank