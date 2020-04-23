#!/usr/bin/env bash

# processes output from batch_run.sh
# todo: concatenate these nicely with paste

# print final accuracy numbers
grep  "Final accuracy" $1 | sed 's/Final accuracy: \+//g'

echo '====='
# print perplexity numbers
grep -B 1 "Best decoding" $1 | grep "=" | sed 's/[+= ]\+/ /g' | sed 's/\^ \+//g'
