#!/bin/bash

# python main.py --data_name=Beauty --model=TVRec \
#                 --lr=0.0005 \
#                 --reg_weight=0 \
#                 --hidden_dropout_prob=0.5 \
#                 --M=32 \
#                 --max_seq_length=50 \
#                 --train_name=TVRec_Beauty_best --save
# wait

# python main.py --data_name=Foursquare --model=TVRec \
#                 --lr=0.0005 \
#                 --reg_weight=1e-05 \
#                 --hidden_dropout_prob=0.2 \
#                 --M=8 \
#                 --max_seq_length=50 \
#                 --train_name=TVRec_FoursquareNYC_best --save
# wait

# python main.py --data_name=LastFM --model=TVRec \
#                 --lr=0.001 \
#                 --reg_weight=1e-3 \
#                 --hidden_dropout_prob=0.4 \
#                 --M=8 \
#                 --max_seq_length=50 \
#                 --train_name=TVRec_LastFM_best --save
# wait

# python main.py --data_name=ML-1M --model=TVRec \
#                 --lr=0.001 \
#                 --reg_weight=1e-05 \
#                 --hidden_dropout_prob=0.3 \
#                 --M=8 \
#                 --max_seq_length=50 \
#                 --train_name=TVRec_ML-1M_best --save
# wait

python main.py --data_name=Sports --model=TVRec \
                --lr=0.0005 \
                --reg_weight=0 \
                --hidden_dropout_prob=0.5 \
                --M=16 \
                --max_seq_length=50 \
                --train_name=TVRec_Sports_best --save
wait

# python main.py --data_name=Yelp --model=TVRec \
#                 --lr=0.0005 \
#                 --reg_weight=1e-3 \
#                 --hidden_dropout_prob=0.1 \
#                 --M=16 \
#                 --max_seq_length=50 \
#                 --train_name=TVRec_Yelp_best --save
# wait


