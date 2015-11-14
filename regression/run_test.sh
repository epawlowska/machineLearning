#!/bin/bash

train_file="train.csv"
test_file="test.csv"
out_file="out.csv"

python data_generator.py --train_file $train_file --test_file $test_file
python main.py --show --train_file train.csv --test_file test.csv -o out.csv -l 0.9 --decay_learning_rate 0.999 --batch_size 0 --n_iter 1000 --shuffle --holdout_size 0.3 --l2 0.0 --rmsprop --hash 1
rm $train_file $test_file $out_file

python data_generator.py --train_file $train_file --test_file $test_file --logistic
python main.py --show --loss logistic --train_file $train_file --test_file $test_file -o $out_file -l 0.079 --decay_learning_rate 0.9999 --n_iter 2000
rm $train_file $test_file $out_file

