# Decrypting the Wilkinson book cipher

[link to paper]

This repo presents a method for solving word-based substitution ciphers with a neural language model.
The source code is in `bookcipher`
`wordbanks` contains wordbanked letters, i.e. the training set




## Installation

    pip install

    # One of the following to install pattern
    # brew install mysql # OSX
    # sudo apt-get install libmysqlclient-dev # ubuntu

    pip3 install pattern


## Running

    python bookcipher/runtime.py --lm gpt2 -b=4
    python bookcipher/runtime.py --help # for more options