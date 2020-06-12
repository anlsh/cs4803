#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality,
#       number of epochs, weigh decay factor, momentum, batch size, learning
#       rate mentioned here to achieve good performance
#############################################################################
# python -u train.py \
#     --model softmax \
#     --epochs 1 \
#     --weight-decay 0.0 \
#     --momentum 0.0 \
#     --batch-size 512 \
#     --lr 0.01 | tee softmax.log
python -u train.py --model softmax --epochs 5 --weight-decay 0.0 --momentum 0.1 --batch-size 512 --lr 0.01 | tee softmax.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
