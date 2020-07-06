import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm

class RandomWalk(object):
    def __init__(self, u_interaction_file, u_tag_file, i_interaction_file, i_tag_file,
                 datas, usize=28745, isize=8090, path_len=5, window_size=2, negative_size=2, prob=0.5):

        #self.users = users
        #self.items = items
        self.datas = datas
        self.usize = usize
        self.isize = isize
        self.path_len = path_len
        self.window_size = window_size
        self.negative_size = negative_size
        self.prob = prob

        self.u_interaction = self.get_neighbours(u_interaction_file)
        if self.prob == 1:
            self.u_tag = self.u_interaction
        else:
            self.u_tag = self.get_neighbours(u_tag_file)

        self.i_interaction = self.get_neighbours(i_interaction_file)
        self.i_tag = self.get_neighbours(i_tag_file)

        #self.u_neg_neighbours, self.i_neg_neighbours = self.get_negsamples()
        self.u_neg_neighbours = {}
        self.i_neg_neighbours = {}
        #self.usamples, self.ulabels, self.isamples, self.ilabels = self.get_samples()

    def get_neighbours(self, path):
        print 'getting neighbours:', path
        neigh = np.loadtxt(path, dtype=int)

        neighbours_dict = defaultdict(list)
        for lis in tqdm(neigh):
            uid = lis[0]
            unid = lis[1]
            if uid not in neighbours_dict:
                neighbours_dict[uid] = []
            neighbours_dict[uid].append(unid)

            if unid not in neighbours_dict:
                neighbours_dict[unid] = []
            neighbours_dict[unid].append(uid)
        return neighbours_dict

    def get_negsamples(self):
        print 'processing randomWalk.get_negsamples'
        u_neg_dict = {}
        for uid in tqdm(range(self.usize)):
            if uid not in self.u_interaction and uid not in self.u_tag:
                pass

            else:
                u_neg_dict[uid] = set()
                if uid not in self.u_interaction:
                    for un in range(self.usize):
                        if un in self.u_tag[uid]:
                            pass
                        else:
                            u_neg_dict[uid].add(un)

                elif uid not in self.u_tag:
                    for un in range(self.usize):
                        if un in self.u_tag[uid]:
                            pass
                        else:
                            u_neg_dict[uid].add(un)

                else:
                    for un in range(self.usize):
                        if un not in self.u_interaction[uid] and un not in self.u_tag[uid]:
                            u_neg_dict[uid].add(un)

        i_neg_dict = {}
        for iid in tqdm(range(self.isize)):
            if iid not in self.i_interaction and iid not in self.i_tag:
                pass

            else:
                i_neg_dict[iid] = set()
                if iid not in self.i_interaction:
                    for iin in range(self.isize):
                        if iin in self.i_tag[iid]:
                            pass
                        else:
                            i_neg_dict[iid].add(iin)

                elif iid not in self.u_tag:
                    for iin in range(self.isize):
                        if iin in self.i_tag[iid]:
                            pass
                        else:
                            i_neg_dict[iid].add(iin)

                else:
                    for iin in range(self.isize):
                        if iin not in self.i_interaction[iid] and iin not in self.i_tag[iid]:
                            i_neg_dict[iid].add(iin)

        return u_neg_dict, i_neg_dict

    def get_samples(self):
        print 'processing randomWalk.get_samples()'
        uisamples = []
        usamples = []

        isamples = []
        ilabels = []


        uc = 0
        um = 0
        ui = 0
        ut = 0
        ic = 0
        im = 0
        ii = 0
        it = 0
        uloop_c = 0
        iloop_c = 0
        for idx in tqdm(range(len(self.datas))):
            unode = self.datas[idx][0]
            inode = self.datas[idx][1]
            label = self.datas[idx][2]
            step = self.datas[idx][3]
            neg_unsamples = []
            neg_insamples = []
            rdm = random.uniform(0, 1)

            #new_pair is generate different center and context word
            unew_pair = []
            inew_pair = []
            # choice which kind of edges
            if unode in self.u_interaction and unode in self.u_tag and \
                            len(self.u_interaction[unode]) > 0 and len(self.u_tag[unode]) > 0:
                um+= 1
                if rdm < self.prob:
                    upos = random.choice(self.u_interaction[unode])

                else:
                    upos = random.choice(self.u_tag[unode])

                if len(self.u_interaction[unode]) >= 2 and len(self.u_tag[unode]) >= 2:
                    urdm = random.uniform(0, 1)
                    if urdm < self.prob:
                        unew_pair = random.sample(self.u_interaction[unode], 2)
                    else:
                        unew_pair = random.sample(self.u_tag[unode], 2)

                elif len(self.u_interaction[unode]) >= 2:
                    unew_pair = random.sample(self.u_interaction[unode], 2)
                elif len(self.u_tag[unode]) >= 2:
                    unew_pair = random.sample(self.u_tag[unode], 2)
                else:
                    urdm = random.uniform(0, 1)
                    if urdm < self.prob:
                        unew_pair = [self.u_interaction[unode][0], unode]
                    else:
                        unew_pair = [self.u_tag[unode][0], unode]

            elif unode in self.u_interaction and len(self.u_interaction[unode]) > 0:
                ui += 1
                upos = random.choice(self.u_interaction[unode])
                if len(self.u_interaction[unode]) >=2 :
                    unew_pair = random.sample(self.u_interaction[unode], 2)
                else:
                    unew_pair = [self.u_interaction[unode][0], unode]

            elif unode in self.u_tag and len(self.u_tag[unode]) > 0:
                ut += 1
                upos = random.choice(self.u_tag[unode])
                if len(self.u_tag[unode]) >=2 :
                    unew_pair = random.sample(self.u_tag[unode], 2)
                else:
                    unew_pair = [self.u_tag[unode][0], unode]

            else:
                uc+= 1
                upos = random.randint(0, self.usize-1)
                unew_pair = [random.randint(0, self.usize-1), unode]

            usamples.append([unode, upos, 1])
            #ulabels.append(1)
            if label == 1:
                uisamples.append([unode, inode, label, step])
            else:
                uisamples.append([unode, inode, 0, step])
            if unode in self.u_neg_neighbours and self.u_neg_neighbours[unode] > 0:
                uneg = random.sample(self.u_neg_neighbours[unode], self.negative_size)
            else:
                #uneg = [random.randint(0, self.usize-1) for i in range(self.negative_size)]

                uneg = []
                loop = 0
                while (len(uneg) < self.negative_size and loop < self.negative_size * 100):
                    nud = random.randint(0, self.usize - 1)
                    if nud not in self.u_interaction[unode] and nud not in self.u_tag[unode]:
                        uneg.append(nud)
                    loop += 1
                while (len(uneg) < self.negative_size):
                    nud = random.randint(0, self.usize - 1)
                    uneg.append(nud)

                if  loop >= self.negative_size * 100 :
                    uloop_c+= 1

            for un in uneg:
                #usamples.append(un)
                #ulabels.append(0)
                usamples.append([unode, un, 0])
                neg_unsamples.append([un, unode, 0])

                if label == 1:
                    uisamples.append([unode, inode, label, step])
                else:
                    uisamples.append([unode, inode, 0, step])


            irdm = random.uniform(0, 1)
            inew_pair = []
            # choice which kind of edges
            if inode in self.i_interaction and inode in self.i_tag and len(self.i_interaction[inode])> 0 and len(self.i_tag[inode])> 0:
                im += 1
                if irdm < self.prob:
                    ipos = random.choice(self.i_interaction[inode])

                else:
                    ipos = random.choice(self.i_tag[inode])

                if len(self.i_interaction[inode]) >= 2 and len(self.i_tag[inode]) >= 2:
                    icrdm = random.uniform(0, 1)
                    if icrdm < self.prob:
                        inew_pair = random.sample(self.i_interaction[inode], 2)
                    else:
                        inew_pair = random.sample(self.i_tag[inode], 2)

                elif len(self.i_interaction[inode]) >= 2:
                    inew_pair = random.sample(self.i_interaction[inode], 2)
                elif len(self.i_tag[inode]) >= 2:
                    inew_pair = random.sample(self.i_tag[inode], 2)
                else:
                    icrdm = random.uniform(0, 1)
                    if icrdm < self.prob:
                        inew_pair = [self.i_interaction[inode][0], inode]
                    else:
                        inew_pair = [self.i_tag[inode][0], inode]

            elif inode in self.i_interaction and len(self.i_interaction[inode])>0:
                ii += 1
                ipos = random.choice(self.i_interaction[inode])
                if len(self.i_interaction[inode]) >=2 :
                    inew_pair = random.sample(self.i_interaction[inode], 2)
                else:
                    inew_pair = [self.i_interaction[inode][0], inode]

            elif inode in self.i_tag and len(self.i_tag[inode])>0:
                it += 1
                ipos = random.choice(self.i_tag[inode])
                if len(self.i_tag[inode]) >=2 :
                    inew_pair = random.sample(self.i_tag[inode], 2)
                else:
                    inew_pair = [self.i_tag[inode][0], inode]

            else:
                ic += 1
                ipos = random.randint(0, self.isize - 1)

                inew_pair = [random.randint(0, self.isize - 1), inode]

            #isamples.append(ipos)
            isamples.append([inode, ipos, 1])
            #ilabels.append(1)

            if inode in self.i_neg_neighbours:
                ineg = random.sample(self.i_neg_neighbours[inode], self.negative_size)
            else:
                #ineg = [random.randint(0, self.isize - 1) for i in range(self.negative_size)]
                ineg = []
                iloop = 0
                while(len(ineg) < self.negative_size and  iloop < self.negative_size *100):

                    nid = random.randint(0, self.isize - 1)
                    if nid not in self.i_interaction[inode] and nid not in self.i_tag[inode]:
                        ineg.append(nid)
                    iloop += 1
                while (len(ineg) < self.negative_size):
                    nid = random.randint(0, self.isize - 1)
                    ineg.append(nid)

                if iloop >= self.negative_size * 100:
                    iloop_c += 1

            for iin in ineg:
                #isamples.append(iin)
                #ilabels.append(0)
                isamples.append([inode, iin, 0])
                neg_insamples.append([iin, inode, 0])

            if len(unew_pair) > 0 and len(inew_pair) > 0:
                usamples.append([unew_pair[0], unew_pair[1], 1])
                isamples.append([inew_pair[0],  inew_pair[1], 1])
                if label == 1:
                    uisamples.append([unode, inode, label, step])
                else:
                    uisamples.append([unode, inode, 0, step])

                for idx, lis in enumerate(neg_unsamples):
                    usamples.append(lis)
                    if label == 1:
                        uisamples.append([unode, inode, label, step])
                    else:
                        uisamples.append([unode, inode, 0, step])

                for ilis in neg_insamples:
                    isamples.append(ilis)


        unode_samples = (np.array(usamples)[:, 0])
        unei_samples = (np.array(usamples)[:, 1])
        ulabels =  (np.array(usamples)[:, 2])
        inode_samples = (np.array(isamples)[:, 0])
        inei_samples = (np.array(isamples)[:, 1])
        ilabels = (np.array(isamples)[:, 2])

        users = (np.array(uisamples)[:, 0])
        items = (np.array(uisamples)[:, 1])
        labels = (np.array(uisamples)[:, 2])
        steps = (np.array(uisamples)[:, 3])
        print 'uc:', uc
        print 'um:', um
        print 'ui:', ui
        print 'ut:', ut
        print 'ic:', ic
        print 'im:', im
        print 'ii:', ii
        print 'it:', it
        print 'uloopc:', uloop_c
        print 'iloopc:', iloop_c



        #return  usamples, ulabels, isamples, ilabels
        return users, items, labels, steps, unode_samples, unei_samples, ulabels, inode_samples, inei_samples, ilabels




class Batcher(object):
    def __init__(self, input_file, u_interaction_file, u_tag_file, i_interaction_file, i_tag_file, usize=904807, isize=393561,
                 batch_size=1024, negative_size=2, prob=0.5, shuffle=True):
        self.batch_size = batch_size
        self.input_file = input_file

        self.negative_size = negative_size
        self.prob = prob
        self.shuffle = shuffle

        #ulis, ilis, self.users, self.items, self.labels = self.read_files()
        self.datas = self.read_files(input_file)

        self.uneighbors = self.read_files_csv(u_interaction_file)

        self.ineighbors = self.read_files_csv(i_interaction_file)

        print 'finish load data'
        #randomWalk = RandomWalk(u_interaction_file, u_tag_file, i_interaction_file, i_tag_file, self.datas, usize=usize, isize=isize,negative_size=self.negative_size, prob=self.prob)

        #self.users, self.items, self.labels, self.steps, self.unode_samples, self.unei_samples, self.ulabels, self.inode_samples, self.inei_samples, self.ilabels = randomWalk.get_samples()

        self.data_size = max(len(self.datas), len(self.uneighbors), len(self.ineighbors))
        self.users, self.items, self.labels, self.steps, self.unode_samples, self.unei_samples, self.ulabels, self.inode_samples, self.inei_samples, self.ilabels = self.generate_samples()

        self.start_index = 0
        if self.shuffle:
            self.shuffle_data()


    def generate_samples(self):

        num_datas = len(self.datas)
        num_uneighbors = len(self.uneighbors)
        num_ineighbors = len(self.ineighbors)
        if  num_datas < self.data_size:
            num_repeat = self.data_size/num_datas
            uidatas = np.tile(self.datas, (num_repeat,1))

            num_choice = self.data_size % num_datas
            perm = np.random.permutation(num_datas)
            uidatas = np.concatenate((uidatas, self.datas[perm[:num_choice]]), axis=0)

        else:
            uidatas = self.datas
        print 'finish getting uidata'
        if num_uneighbors < self.data_size:
            num_repeat = self.data_size / num_uneighbors
            uneighbors = np.tile(self.uneighbors, (num_repeat, 1))

            num_choice = self.data_size % num_uneighbors
            perm = np.random.permutation(num_uneighbors)
            uneighbors = np.concatenate((uneighbors, self.uneighbors[perm[:num_choice]]), axis=0)
        else:
            uneighbors = self.uneighbors
        print 'finish getting uneighbors'

        if num_ineighbors < self.data_size:
            num_repeat = self.data_size / num_ineighbors
            ineighbors = np.tile(self.ineighbors, (num_repeat, 1))

            num_choice = self.data_size % num_ineighbors
            perm = np.random.permutation(num_ineighbors)
            ineighbors = np.concatenate((ineighbors, self.ineighbors[perm[:num_choice]]), axis=0)
        else:
            ineighbors = self.ineighbors
        print 'finish getting ineighbors'
        return uidatas[:, 0], uidatas[:, 1], uidatas[:, 2], uidatas[:, 3], uneighbors[:, 0], uneighbors[:, 1], uneighbors[:, 2], ineighbors[:, 0], ineighbors[:,1], ineighbors[:,2]



    def ts(self):
        print '--------------------data size-----------------------'
        print 'users:', len(self.users)
        print 'items:', len(self.items)
        print 'labels:', len(self.labels)
        print 'unode_samples:', len(self.unode_samples)
        print 'unei_samples:', len(self.unei_samples)
        print 'ulabels:', len(self.ulabels)
        print 'inode_samples:', len(self.inode_samples)
        print 'inei_samples:', len(self.inei_samples)
        print 'ilabels:', len(self.ilabels)
        print '-----------------------------------------------------'

        print '------------------top 6 data -----------------------'
        print 'users:', self.users[:10]
        print 'items:', self.items[:10]
        print 'labels:', self.labels[:10]
        print 'unode_samples:', self.unode_samples[:10]
        print 'unei_samples:', self.unei_samples[:10]
        print 'ulabels:', self.ulabels[:10]
        print 'inode_samples:', self.inode_samples[:10]
        print 'inei_samples:', self.inei_samples[:10]
        print 'ilabels:', self.ilabels[:10]
        print '*******************************************************'

        count_labels = 0
        count_ilabels = 0
        count_ulabels = 0
        for idx, v in enumerate(self.labels):
            if v != 0 and v!= 1:
                count_labels += 1
                print 'v: ', v

        for vi in self.ilabels:
            if vi != 0 and vi!= 1:
                count_ilabels += 1
                print 'vi: ', vi

        for vu in self.ulabels:
            if vu != 0 and vu!= 1:
                count_ulabels += 1
                print 'vu: ', vu
        print '+++++++++++++++++++++++++++++++++++++++++++++++'
        print 'count_labels:  ', count_labels , '  count_ilabels: ', count_ilabels , '   count_ulabels:', count_ulabels


    def shuffle_data(self):
        """
               Shuffles maintaining the same order.
               """
        perm = np.random.permutation(self.data_size)  # perm of index in range(0, data_size)
        assert len(perm) == self.data_size
        self.users, self.items, self.labels, self.steps, self.unode_samples, self.unei_samples, self.ulabels, self.inode_samples,  self.inei_samples, self.ilabels = \
                                                        self.users[perm], self.items[perm], self.labels[perm], self.steps[perm], \
                                                        self.unode_samples[perm], self.unei_samples[perm], self.ulabels[perm], \
                                                        self.inode_samples[perm], self.inei_samples[perm], self.ilabels[perm]

    def get_next_batch(self):
        """
        returns the next batch
        """
        while True:
            if self.start_index >= self.data_size:

                self.start_index = 0
                if self.shuffle:
                    self.shuffle_data()
            else:
                num_data_returned = min(self.batch_size, self.data_size - self.start_index)
                assert num_data_returned > 0
                end_index = self.start_index + num_data_returned

                yield  self.users[self.start_index:end_index], self.items[self.start_index:end_index], self.labels[self.start_index:end_index], self.steps[self.start_index:end_index], \
                       self.unode_samples[self.start_index:end_index], self.unei_samples[self.start_index:end_index], self.ulabels[self.start_index:end_index], \
                       self.inode_samples[self.start_index:end_index], self.inei_samples[self.start_index:end_index], self.ilabels[self.start_index:end_index]
                self.start_index = end_index


    def read_files(self, input_file):
        data = np.loadtxt(input_file, dtype=int)
        '''
        users = data[:, 0]
        items = data[:, 1]
        #labels = data[:, 2]
    
        merge_users = []
        merge_items = []
        merge_labels = []
        for lis in tqdm(data):
            user = lis[0]
            item = lis[1]
            label = lis[2]
            for i in range(self.negative_size+1):
                merge_users.append(user)
                merge_items.append(item)
                if label == 1:

                    merge_labels.append(label)
                else:
                    merge_labels.append(0)
        merge_users = np.array(merge_users)
        merge_items = np.array(merge_items)
        merge_labels = np.array(merge_labels)
        print 'finish read data'
        return users, items, merge_users, merge_items, merge_labels
        '''
        print 'finish read data'
        return data

    def read_files_csv(self, input_file):
        data = np.loadtxt(input_file, dtype=int, delimiter=',')
        return data

    def write_to_file(self, dat, path):
        with open(path, 'w') as f:
            for lis in dat:
                for w in lis:
                    f.write(str(w) + ' ')
                f.write('\n')
        print 'finish write to file:', path

    def save(self):
        data_size = len(self.users)
        with open('random_walk/user_item_pairs.txt', 'w') as f:
            for idx in range(data_size):
                f.write(str(self.users[idx]) + ' '+ str(self.items[idx])+ ' ' + str(self.labels[idx])+ ' ' + str(self.steps[idx]) + '\n')


        with open('random_walk/user_pairs.txt', 'w') as f:
            for idx in range(data_size):
                f.write(str(self.unode_samples[idx]) + ' '+ str(self.unei_samples[idx])+ ' ' + str(self.ulabels[idx])+ '\n')

        with open('random_walk/item_pairs.txt', 'w') as f:
            for idx in range(data_size):
                f.write(str(self.inode_samples[idx]) + ' ' + str(self.inei_samples[idx]) + ' ' + str(self.ilabels[idx]) + '\n')





if __name__=='__main__':
    train_file = '../data_processed/Book/train_user_item_label_interaction_step.txt'
    u_interaction_file = '../data_processed/Book/user_neighbors_by_items_avg_ratio.txt'

    u_tag_file = '../data_processed/Book/user_neighbors_by_items_avg_ratio.txt'

    i_interaction_file = '../data_processed/Book/item_item_similar.txt'
    i_tag_file = '../data_processed/Book/item_item_similar.txt'

    BC = Batcher(train_file, u_interaction_file, u_tag_file, i_interaction_file, i_tag_file, shuffle=False)
    #BC.ts()
    BC.save()