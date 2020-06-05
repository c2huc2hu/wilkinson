#!/usr/bin/env bash
# if block_size isn't specified, we get an error about num_samples=0 on small datasets

export CUDA_VISIBLE_DEVICES=0
block_size=512

# COFEA
# epochs = 1 3 10 20
# corpus = 'cofea'
# train_file = 'from-website-uncased.txt'

# ORACLE
declare -a epochs=(3 10 50 200)
corpus='oracle'
train_file='decoding.txt'

# WILKINSON
declare -a epochs=(1 3 10 20)
corpus='wilkinson'
train_file='wilkinson_letters.txt'

for epoch in ${epochs[@]}
do
    echo $epoch

    mkdir  -p language-models/$corpus/epoch-$epoch
    python3.7 ~/transformers/examples/language-modeling/run_language_modeling.py \
        --output_dir=language-models/$corpus/epoch-$epoch --model_name_or_path=gpt2 \
        --do_train --train_data_file=language-models/$corpus/$train_file \
        --do_eval --eval_data_file=data/eval/plaintext.txt --overwrite_output_dir \
        --block_size=$block_size --num_train_epochs=$epoch 2>&1 language-models/$corpus/epoch-$epoch/eval.out | 
        tee language-models/$corpus/epoch-$epoch/train.out

    python3.7 ~/transformers/examples/language-modeling/run_language_modeling.py \
        --output_dir=language-models/$corpus/epoch-$epoch --model_name_or_path=language-models/$corpus/epoch-$epoch \
        --do_eval --eval_data_file=data/eval/plaintext.txt --overwrite_output_dir --block_size=$block_size 2>&1 |
        tee language-models/$corpus/epoch-$epoch/eval.out

    python3 bookcipher/runtime.py --lm language-models/$corpus/epoch-$epoch -b 16  2>&1 | tee language-models/$corpus/epoch-$epoch/decoding-b16.out
done
