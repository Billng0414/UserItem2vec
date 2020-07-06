# UserItem2vec
code of UserItem2vec

# Multi-context aware user-item embedding for recommendation
## This is the python implementation -- "Multi-context aware user-item embedding for recommendation".


# Requirements
python2.7
tensorflow == 0.12
numpy
tqdm


# Note
## to run this code, you need to prepare the sampled data and:
1. set your data set  "train_file" in line 460 in code/train.py 
2. set user-user edge  "u_interaction_file" in line 443 in code/train.py
3. set item-item edge  "i_interaction_file" in line 451 in code/train.py

## the format of train_file is :
user item label step  
(step is the order of user item interaction, and if you don't care about higher-order interactions, set this to 1)

## the format of  u_interaction_file/ i_interaction_file is:
node1,node2,label 


## you can also use our sampling algorithm and you need to:
1.  set your data set  "train_file" 
2.  set user-user edge  "u_interaction_file" and "u_tag_file" 
3.  set item-item edge "i_interaction_file" and "i_tag_file" 
4.  change "from feed_data_out_rwd import Batcher" to "from feed_data import Batcher" in line 5 in code/train.py

## the options description 
--user_embedding_size  embedding size
--negative_size   negative sample
--probability     the probability to balance different kind of user-user/item-item edge
--learning_rate   learning rate
--alpha           the weight of the loss


