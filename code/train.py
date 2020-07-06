import tensorflow as tf
import numpy as np
import time
import os
from feed_data_out_rwd import Batcher
import argparse 

def load_datas(fname, dtype=float):
    '''
    for each row, the first index is nid and the values are separate by whitespace.
    return (datas, datas_id): in datas, the nid are discard and responsed by index. in datas_id, it is the original data.
    '''
    datas_id = np.loadtxt(fname, dtype)
    assert set(range(datas_id.shape[0])) == set(datas_id[:, 0])

    shape = (datas_id.shape[0], datas_id.shape[1] - 1)
    idx = np.array(datas_id[:, 0], dtype=int)
    datas = np.zeros(shape, dtype)
    datas[idx, :] = datas_id[:, 1:]
    datas = np.array(datas)
    return (datas, datas_id)


class Trainer(object):
    def __init__(self):


        if load_pretrained_vectors:

            print('pretrained entity & word embeddings available. Initializing with them.')

            pretrained_user_vectors, _  = load_datas(pretrained_user_vectors_path)
            pretrained_item_vectors, _ = load_datas(pretrained_item_vector_path)

            assert (pretrained_user_vectors is not None)

            initializer_op_user = tf.contrib.layers.xavier_initializer()
            initializer_op_item = tf.contrib.layers.xavier_initializer()
            initializer_op_user_center =  tf.contrib.layers.xavier_initializer()
            initializer_op_item_center =  tf.contrib.layers.xavier_initializer()
            initializer_op_user_context = tf.constant_initializer(pretrained_user_vectors)
            initializer_op_item_context = tf.constant_initializer(pretrained_item_vectors)

        else:
            print('No pretrained entity & word embeddings available. Learning entity embeddings from scratch')

            initializer_op_user = tf.contrib.layers.xavier_initializer()
            initializer_op_item = tf.contrib.layers.xavier_initializer()
            initializer_op_user_center = tf.contrib.layers.xavier_initializer()
            initializer_op_item_center = tf.contrib.layers.xavier_initializer()
            initializer_op_user_context = tf.contrib.layers.xavier_initializer()
            initializer_op_item_context = tf.contrib.layers.xavier_initializer()


        self.user_lookup_table = tf.get_variable("user_lookup_table",
                                                 shape=[user_size, user_embedding_size],
                                                 dtype=tf.float32,
                                                 initializer=initializer_op_user,
                                                 trainable=True)

        self.user_lookup_table_context = tf.get_variable("user_lookup_table_context",
                                                         shape=[user_size, user_embedding_size],
                                                         dtype=tf.float32,
                                                         initializer=initializer_op_user_context,
                                                         trainable=False)

        self.user_lookup_table_center = tf.get_variable("user_lookup_table_center",
                                                        shape=[user_size, user_embedding_size],
                                                        dtype=tf.float32,
                                                        initializer=initializer_op_user_center,
                                                        trainable=True)


        self.item_lookup_table = tf.get_variable("item_lookup_table",
                                                 shape=[item_size, item_embedding_size],
                                                 dtype=tf.float32,
                                                 initializer=initializer_op_item,
                                                 trainable=True)

        self.item_lookup_table_context = tf.get_variable("item_lookup_table_context",
                                                         shape=[item_size, item_embedding_size],
                                                         dtype=tf.float32,
                                                         initializer=initializer_op_item_context,
                                                         trainable=False)

        self.item_lookup_table_center = tf.get_variable("item_lookup_table_center",
                                                        shape=[item_size, item_embedding_size],
                                                        dtype=tf.float32,
                                                        initializer=initializer_op_item_center,
                                                        trainable=True)


        self.batcher = Batcher(train_file, u_interaction_file, u_tag_file, i_interaction_file, i_tag_file, usize=user_size, isize=item_size,
                               batch_size=batch_size, negative_size=negative_size, prob=probability, shuffle=True)

        #self.optimizer = tf.train.AdamOptimizer(learning_rate)


    def bp(self, cost):
        tvars = tf.trainable_variables()
        grads = tf.gradients(cost, tvars)
        grads, _ = tf.clip_by_global_norm(grads, grad_clip_norm)
        #train_op = self.optimizer.apply_gradients(zip(grads, tvars))

        if gradient_version == 'SGD':
            train_op =  tf.train.GradientDescentOptimizer(learning_rate).apply_gradients(zip(grads, tvars))

        elif gradient_version == 'Momentum':
            train_op = tf.train.MomentumOptimizer(learning_rate,  momentum=0.9).apply_gradients(zip(grads, tvars))

        elif gradient_version == 'AdaDelta':
            train_op = tf.train.AdadeltaOptimizer(learning_rate).apply_gradients(zip(grads, tvars))

        else:
            train_op = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(grads, tvars))

        return train_op

    '''
    def bp(self, cost):
        tvars = tf.trainable_variables()
        grads = tf.gradients(cost, tvars)
        grads, _ = tf.clip_by_global_norm(grads, grad_clip_norm)
        train_op = self.optimizer.apply_gradients(zip(grads, tvars))
        return train_op
    '''

    def initialize(self):
        self.user = tf.placeholder(tf.int32, shape=[None], name='user')
        self.item = tf.placeholder(tf.int32, shape=[None], name='item')
        #self.timeslot = tf.placeholder(tf.int32, shape=[None], name='timeslot')
        self.label = tf.placeholder(tf.float32, shape=[None], name='label')
        self.step = tf.placeholder(tf.int32, shape=[None], name='step')

        self.unode = tf.placeholder(tf.int32, shape=[None], name='unode')
        self.uneighbour = tf.placeholder(tf.int32, shape=[None], name='uneighbour')

        self.inode = tf.placeholder(tf.int32, shape=[None], name='inode')
        self.ineighbour = tf.placeholder(tf.int32, shape=[None], name='ineighbour')

        self.ulabel = tf.placeholder(tf.float32, shape=[None], name='ulabel')
        self.ilabel = tf.placeholder(tf.float32, shape=[None], name='ilabel')



        user_embedding = tf.nn.embedding_lookup(self.user_lookup_table, self.user, name="user_embedding")
        item_embedding =  tf.nn.embedding_lookup(self.item_lookup_table, self.item, name="item_embedding")

        unode_embedding = tf.nn.embedding_lookup(self.user_lookup_table, self.unode, name="unode_embedding")
        user_neighbour_embedding = tf.nn.embedding_lookup(self.user_lookup_table_center, self.uneighbour,
                                                          name='user_neighbour_embeeding')

        unei_context_embedding = tf.nn.embedding_lookup(self.user_lookup_table_context, self.uneighbour,
                                                        name='unei_context_embedding')

        inode_embedding = tf.nn.embedding_lookup(self.item_lookup_table, self.inode, name="inode_embedding")
        item_neighbour_embedding = tf.nn.embedding_lookup(self.item_lookup_table_center, self.ineighbour,
                                                          name='item_neighbour_embedding')

        inei_context_embedding = tf.nn.embedding_lookup(self.item_lookup_table_context, self.ineighbour,
                                                        name='inei_context_embedding')

        #calculate cosine similarity (dot product)

        cos_similarity = tf.multiply(user_embedding, item_embedding)

        cos_similarity_u = tf.multiply(unode_embedding, user_neighbour_embedding)
        cos_similarity_uc = tf.multiply(unode_embedding, unei_context_embedding)


        cos_similarity_i = tf.multiply(inode_embedding, item_neighbour_embedding)
        cos_similarity_ic = tf.multiply(inode_embedding, inei_context_embedding)

        cos_value = tf.reduce_sum(cos_similarity, 1)

        #stepweight = tf.reciprocal(self.step, name='reciprocal')
        stepweight = tf.exp(tf.to_float(tf.negative(self.step)), name='stepweight')
        #stepweight = 1.0 / self.step
        cos_value_weight = stepweight * cos_value

        logit = tf.nn.sigmoid_cross_entropy_with_logits(cos_value_weight, self.label)


        cos_value_u = tf.reduce_sum(cos_similarity_u, 1)
        logit_u = tf.nn.sigmoid_cross_entropy_with_logits(cos_value_u, self.ulabel)

        cos_value_uc = tf.reduce_sum(cos_similarity_uc, 1)
        logit_uc = tf.nn.sigmoid_cross_entropy_with_logits(cos_value_uc, self.ulabel)

        cos_value_i = tf.reduce_sum(cos_similarity_i, 1)
        logit_i = tf.nn.sigmoid_cross_entropy_with_logits(cos_value_i, self.ilabel)

        cos_value_ic = tf.reduce_sum(cos_similarity_ic, 1)
        logit_ic = tf.nn.sigmoid_cross_entropy_with_logits(cos_value_ic, self.ilabel)

        self.uloss = tf.reduce_sum(logit_u)
        self.ucloss = tf.reduce_sum(logit_uc)
        self.iloss = tf.reduce_sum(logit_i)
        self.icloss = tf.reduce_sum(logit_ic)
        self.merge_uloss = tf.add(self.uloss, self.ucloss)
        self.merge_iloss = tf.add(self.iloss, self.icloss)

        self.merge_loss = tf.add(self.merge_uloss, self.merge_iloss)

        self.uiloss = tf.reduce_sum(logit)


        self.merge_loss = tf.add(self.merge_uloss, self.merge_iloss)

        self.uiloss = alpha * self.uiloss

        self.loss = tf.add(self.merge_loss, self.uiloss) + lamda * tf.nn.l2_loss(self.user_lookup_table)  + \
                    lamda * tf.nn.l2_loss(self.item_lookup_table)


        self.saver = tf.train.Saver()
        self.train_op = self.bp(self.loss)

        #self.train_op_choice = self.bp_choice(self.loss)

        init_op = tf.global_variables_initializer()

        return init_op

    def fit(self):

        log_file =  open(out_dir+'loss.log', 'a')
        train_loss = 0
        des_value = 0
        batch_counter = 0
        self.start_time = time.time()

        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            sess.run(self.initialize())
            epcho = 0
            for data in self.batcher.get_next_batch():
                epcho += 1
                batch_counter += 1

                #batch_user, batch_item, batch_label, batch_unode, batch_uneighbour, \
                #batch_ulabel, batch_inode, batch_ineighbour, batch_ilabel = data

                batch_user, batch_item, batch_label, batch_step, \
                batch_unode, batch_uneighbour, batch_ulabel, batch_inode, batch_ineighbour, batch_ilabel  = data

                feed_dict = {self.user: batch_user,
                             self.item: batch_item,
                             self.label: batch_label,
                             self.step: batch_step,
                             self.unode: batch_unode,
                             self.uneighbour: batch_uneighbour,
                             self.ulabel: batch_ulabel,
                             self.inode: batch_inode,
                             self.ineighbour: batch_ineighbour,
                             self.ilabel: batch_ilabel
                             }

                batch_loss_value, batch_uloss, batch_ucloss, batch_iloss, batch_icloss, batch_uiloss, _ = sess.run(
                    [self.loss,
                     self.uloss,
                     self.ucloss,
                     self.iloss,
                     self.icloss,
                     self.uiloss,
                     self.train_op],
                    feed_dict=feed_dict)

                des_value = des_value - batch_loss_value
                batch_loss_value = 1.0 * batch_loss_value
                batch_uloss = 1.0 * batch_uloss
                batch_ucloss = 1.0 * batch_ucloss
                batch_iloss = 1.0 * batch_iloss
                batch_icloss = 1.0 * batch_icloss
                batch_uiloss = 1.0 * batch_uiloss

                print('\t at iter {0:10d} at time {1:10.4f}s train loss: {2:10.4f}  des_value:{3:10.4f}  '
                      'uloss_value:{4:10.4f}  ucloss_value:{5:10.4f}   iloss_value:{6:10.4f}  '
                      ' batch_icloss:{7:10.4f}   uiloss_value:{8:10.4f}    '.format(
                    batch_counter,
                    time.time() - self.start_time,
                    batch_loss_value,
                    des_value,
                    batch_uloss,
                    batch_ucloss,
                    batch_iloss,
                    batch_icloss,
                    batch_uiloss))

                log_file.write('\t at iter {0:10d} at time {1:10.4f}s train loss: {2:10.4f}  des_value:{3:10.4f}  '
                      'uloss_value:{4:10.4f}  ucloss_value:{5:10.4f}   iloss_value:{6:10.4f}  '
                      ' batch_icloss:{7:10.4f}   uiloss_value:{8:10.4f}    '.format(
                    batch_counter,
                    time.time() - self.start_time,
                    batch_loss_value,
                    des_value,
                    batch_uloss,
                    batch_ucloss,
                    batch_iloss,
                    batch_icloss,
                    batch_uiloss))

                if batch_counter % save_counter == 0:
                    save_user_path = out_dir +'user_d' + str(user_embedding_size) + '_bc' + str(int(batch_counter/save_counter)) + '.embs'
                    np.savetxt(save_user_path, self.user_lookup_table.eval(session=sess), fmt='%f')

                    save_item_path = out_dir + 'item_d' + str(item_embedding_size) + '_bc' + str(int(batch_counter/save_counter)) + '.embs'
                    np.savetxt(save_item_path, self.item_lookup_table.eval(), fmt='%f')
                    #save_path = self.saver.save(sess, "/out.ckpt")
                    #print("Saved model to path")


                if  epcho > max_epcho:
                    break
             
                des_value = batch_loss_value


def write_params():
    with open( out_dir + 'params.txt', 'w') as fout:

        #fout.write('DATA_SET: ' + str(DATA_SET) + '\n')
        #fout.write('train_file: ' + str(args.train_file) + '\n')
        fout.write('train_file: ' + str(train_file) + '\n')
        fout.write('u_interaction_file: ' + str(u_interaction_file) + '\n')
        fout.write('u_tag_file: ' + str(u_tag_file) + '\n')
        fout.write('i_interaction_file: ' + str(i_interaction_file) + '\n')
        fout.write('i_tag_file: ' + str(i_tag_file) + '\n')
        fout.write('user_size: ' + str(user_size) + '\n')

        fout.write('interaction_orders: ' + str(args.korder) + '\n')
        fout.write('user_embedding_size: ' + str(args.user_embedding_size) + '\n')

        fout.write('item_embedding_size: ' + str(args.item_embedding_size) + '\n')
        fout.write('batch_size: ' + str(args.batch_size) + '\n')
        fout.write('save_counter: ' + str(args.save_counter) + '\n')
        fout.write('negative_size: ' + str(args.negative_size) + '\n')
        fout.write('probability: ' + str(args.probability) + '\n')
        fout.write('learning_rate: ' + str(args.learning_rate) + '\n')
        fout.write('lambda:' + str(args.lamda) + '\n')
        fout.write('korder: ' + str(args.korder) + '\n')
        fout.write('alpha: ' + str(args.alpha) + '\n')
        fout.write('ratio: ' + str(args.ratio) + '\n')
        fout.write('gradient_version: ' + str(gradient_version) + '\n')
        fout.write('load_pretrained_vectors: ' + str(load_pretrained_vectors) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #parser.add_argument("-ds", "--DATA_SET", required=True)

    #parser.add_argument("-t", "--train_file", required=True)

    #parser.add_argument("-u", "--u_interaction_file", required=True)
    #parser.add_argument("-ut", "--u_tag_file", required=True)
    #parser.add_argument("-i", "--i_interaction_file", required=True)
    #parser.add_argument("-it", "--i_tag_file", required=True)
    parser.add_argument("-k", "--korder", required=True)
    parser.add_argument("--user_size", required=True)
    parser.add_argument("--item_size", required=True)
    parser.add_argument("-lr", "--learning_rate", required=True)
    parser.add_argument("-b", "--batch_size", default=64)
    parser.add_argument("--max_epcho", default=300000)
    parser.add_argument("--save_counter", default=5000)

    parser.add_argument("-ns", "--negative_size", default=5)
    parser.add_argument("-a", "--alpha", default=1)
    parser.add_argument("-r", "--ratio", default=0)
    parser.add_argument("-ud", "--user_embedding_size", default=128)
    parser.add_argument("-id", "--item_embedding_size", default=128)
    parser.add_argument("-p", "--probability", default=0.5)
    parser.add_argument("--grad_clip_norm", default=5)
    parser.add_argument("--gradient_version", default='ADAM')
    parser.add_argument("--debug_mode", default=False)

    parser.add_argument("--lamda", default=0.01)

    parser.add_argument("--load_pretrained_vectors", default=False)



    args = parser.parse_args()

    #DATA_SET = (args.DATA_SET)[:-1]

    #train_file = args.train_file
    #u_interaction_file = args.u_interaction_file
    #u_tag_file = args.u_tag_file
    #i_interaction_file = args.i_interaction_file
    #i_tag_file = args.i_tag_file

    #gradient_version = int(args.gradient_version)
    gradient_version = args.gradient_version
    print 'gradient_version:', gradient_version
    if gradient_version[:-1] == 'SGD':
        gradient_version = 'SGD'
        print 'gradient_version initial as SGD'
    elif gradient_version[:-1] == 'Momentum':
        gradient_version = 'Momentum'
        print 'gradient_version initial as Momentum'

    elif gradient_version[:-1] == 'AdaDelta':
        gradient_version = 'AdaDelta'
        print 'gradient_version initial as AdaDelta'
    else:
        gradient_version = 'ADMM'
        print 'gradient_version initial as ADMM'

    print 'final gradient_version:', gradient_version

    user_size = int(args.user_size)
    item_size = int(args.item_size)

    korder = int(args.korder)
    user_embedding_size = int(args.user_embedding_size)
    item_embedding_size = int(args.item_embedding_size)

    batch_size = int(args.batch_size)
    max_epcho = int(args.max_epcho)
    save_counter = int(args.save_counter)
    negative_size = int(args.negative_size)

    learning_rate = float(args.learning_rate)
    lamda = float(args.lamda)
    alpha = float(args.alpha)
    probability = float(args.probability)
    ratio = float(args.ratio)

    grad_clip_norm = int(args.grad_clip_norm)
    debug_mode = bool(args.debug_mode)
    load_pretrained_vectors = bool(args.load_pretrained_vectors)
    load_pretrained_vectors = 0



    print 'user size:', user_size
    print 'item size:', item_size
    #SGD, ADAM , AdaDelta, Momentum

    data_dir = 'data/'

    pretrained_user_vectors_path = data_dir + ''

    u_interaction_file = data_dir + 'user_save_pairs_r1s5w1n1.csv'

    u_tag_file =  ''

    #for speedup
    #if probability == 1.0:
    #    u_tag_file = data_dir + 'user_friendship.txt'

    i_interaction_file = data_dir + 'item_save_pairs_r3s5w1n1.csv'

    i_tag_file =  ''

    # for speedup
    #if probability == 1.0:
    #    i_tag_file = data_dir + 'item_neigh_by_category.txt'

    if ratio == 0:
        train_file = data_dir + 'train_user_item_label_interaction_step_r0.txt'
    else:
        train_file = data_dir + 'train_user_item_label_interaction_step.txt'

    print 'train_file:', train_file
    path =  'D' + str(user_embedding_size) + 'N' + str(negative_size) + 'P' + str(int(10 * probability)) + \
           'R' + str(int(ratio)) + 'A' + str(alpha) + 'M' + str(100*lamda)+ 'LR' + str(10000 * learning_rate) + 'K' + str(korder) + '_' + \
           str(gradient_version) + '_Version1'

    out_dir = 'results/' + str(path) + '/'

    if debug_mode == True:
        nt = time.strftime("%Y%m%d_%H%H%S", time.localtime(time.time()))
        out_dir = 'results/' +  '/debug_' + str(nt) + '/'

        save_counter = 1000000

    if os.path.exists(out_dir):
        print 'dir is already exist:', out_dir
    else:
        os.mkdir(out_dir)
        print 'sucuess mkdir:', out_dir

    write_params()

    t = Trainer()
    t.fit()
