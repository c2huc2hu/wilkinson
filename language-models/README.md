Fine tuning a language model
```
python3 ../transformers/examples/run_lm_finetuning.py --output_dir=language-models/gold/model3 --model_type=gpt2 --model_name_or_path=/nfs/cold_project/users/chrischu/data/pytorch-transformers/gpt2 \
    --do_train --train_data_file=/nfs/cold_project/users/chrischu/wilkinson/language-models/gold/gold-text.txt \
    --do_eval --eval_data_file=$chrischu/wilkinson/language-models/gold/gold-text.txt --block_size 512 \
    --num_train_epochs=100 --evaluate_during_training --eval_all_checkpoints

```
