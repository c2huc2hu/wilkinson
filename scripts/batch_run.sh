#!/usr/bin/env bash

# run with ./batch_run.sh > batch_runs-`isodate` 2> times.out 

echo "Creating partial files"

head -n 11 data/eval/ciphertext.txt > data/eval/ciphertext-first11.txt
head -n 11 data/eval/plaintext.txt > data/eval/plaintext-first11.txt
head -n 22 data/eval/ciphertext.txt > data/eval/ciphertext-first22.txt
head -n 22 data/eval/plaintext.txt > data/eval/plaintext-first22.txt

python3 bookcipher/runtime.py --lm oracle -b 1
echo "==================== Done run #1 ========================================"

time python3 bookcipher/runtime.py --lm gpt2 -b 1 --source_file data/eval/ciphertext-first11.txt --gold_file data/eval/plaintext-first11.txt
echo "==================== Done run #2a ========================================"
time python3 bookcipher/runtime.py --lm gpt2 -b 1 --source_file data/eval/ciphertext-first22.txt --gold_file data/eval/plaintext-first22.txt
echo "==================== Done run #2b ========================================"
time python3 bookcipher/runtime.py --lm gpt2 -b 1 --source_file data/eval/ciphertext.txt --gold_file data/eval/plaintext.txt
echo "==================== Done run #2c ========================================"


time python3 bookcipher/runtime.py --lm gpt2 -b 4 --source_file data/eval/ciphertext-first11.txt --gold_file data/eval/plaintext-first11.txt
echo "==================== Done run #3a ========================================"
time python3 bookcipher/runtime.py --lm gpt2 -b 4 --source_file data/eval/ciphertext-first22.txt --gold_file data/eval/plaintext-first22.txt
echo "==================== Done run #3b ========================================"
time python3 bookcipher/runtime.py --lm gpt2 -b 4 --source_file data/eval/ciphertext.txt --gold_file data/eval/plaintext.txt
echo "==================== Done run #3c ========================================"


time python3 bookcipher/runtime.py --lm gpt2 -b 16 --source_file data/eval/ciphertext-first11.txt --gold_file data/eval/plaintext-first11.txt
echo "==================== Done run #4a ========================================"
time python3 bookcipher/runtime.py --lm gpt2 -b 16 --source_file data/eval/ciphertext-first22.txt --gold_file data/eval/plaintext-first22.txt
echo "==================== Done run #4b ========================================"
time python3 bookcipher/runtime.py --lm gpt2 -b 16 --source_file data/eval/ciphertext.txt --gold_file data/eval/plaintext.txt
echo "==================== Done run #4c ========================================"



echo "Done all"
