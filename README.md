# Decrypting the Wilkinson book cipher

[link to paper]

This repo presents a method for solving word-based substitution ciphers with a neural language model.

- `bookcipher` - contains source code
- `data` - transcribed real wilkinson data + synthetic data files
- `language-models` - instructions for fine-tuning language models
- `lattices` - test files
- `scripts` - batch run scripts
- `tools` - tools, for now, just the scraper for wilkinson text
- `wordbanks` - contains wordbanked letters, i.e. the training set


## Installation

    pip install -r  requirements.txt

    # One of the following to install pattern
    # brew install mysql # OSX
    # sudo apt-get install libmysqlclient-dev # ubuntu

    pip3 install pattern


## Running the final configuration (with normal GPT, not fine-tuned)

    # no self-learning, fastest
    python bookcipher/runtime.py --lm gpt2 -b 4 --source_file data/eval/ciphertext.txt --gold_file data/eval/plaintext.txt

    # with self-learning, more accurate
    python bookcipher/runtime.py --lm gpt2 -b 4 --source_file data/eval/ciphertext.txt --gold_file data/eval/plaintext.txt --self-learn --confidence_model left

    python bookcipher/runtime.py --help # for more options


## Running synthetic data experiments
    
    # generate and train on the first 800 words of the synthetic corpus
    python bookcipher/encipher.py 800

    # edit wordbank.py MIN_WORD and MAX_WORD to 1, (size of dictionary), respectively
    # yes, this is terrible, sorry
    python bookcipher/runtime.py -b 4  --gold_file data/synthetic/test-plaintext.txt --source_file data/synthetic/test-ciphertext-800.txt  --lm gpt2  --confidence_model left --synthetic --wordbank data/synthetic/wordbank-800 | tee synthetic-800.out
