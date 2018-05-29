import threading
from time import sleep

import numpy as np
import h5py
import cPickle as pickle


import os.path as osp
import time
from util import Config


class Reader(threading.Thread):
    
    def __init__(self, file_data, file_info, config=None):
        
#        Initialize the reader.
        self.feat_dim = config.get("feat_dim")
        self.freq_normalize = freq_normalize = config.get("freq_normalize") # bool, if True, freq-wise normalize, else, global normalize
        self.patch_len = patch_len = config.get("patch_len")
        self.patch_hop = patch_hop = config.get("patch_hop")
        self.rng_seed = rng_seed = config.get("rng_seed")
        self.batch_size = config.get("batch_size")
        self.num_track = config.get("num_track")

        # Initialization of super class
        threading.Thread.__init__(self)

        # Load data and info
        self.f = h5py.File(file_data)
        self.info = info = pickle.load(open(file_info, 'r'))['info']



        # Load the mean and std for normalization
        if freq_normalize:
            data_mean = np.array(self.f['freq_mean'])
            self.data_mean = data_mean[None, :]

            data_std = np.array(self.f['freq_std'])
            self.data_std = data_std[None, :]
        else:
            self.data_mean = np.array(self.f['global_mean'])
            self.data_std = np.array(self.f['global_std'])

        self.rng = np.random.RandomState(seed=rng_seed)


        """ Step 3: Prepare data flow: a list of [start_index, end_index] 
        """
        data_flow = []
        for train_file in info.keys():
            cur_index_start, cur_len = info[train_file]
            if cur_len <= patch_len:
                data_flow.append([cur_index_start, cur_index_start + cur_len])
            else:
                cur_num_patches = (cur_len - patch_len) / patch_hop + 1

                for i in range(cur_num_patches):
                    data_flow.append([cur_index_start + i * patch_hop,
                                           cur_index_start + i * patch_hop + patch_len])
                if ((cur_len - patch_len) % patch_hop) != 0:
                    data_flow.append([cur_index_start + cur_len - patch_len, cur_index_start + cur_len])

        self.rng.shuffle(data_flow)
        self.data_flow = data_flow
        
        """ Step 4. Initialize threading
        """
        self.running = True
        self.data_buffer = None
        self.lock = threading.Lock()
        self.index_start = 0
        self.start()

    def reset(self):
        """ This function is specifically for evaluation.
        :return:
        """
        self.index_start = 0

    def run(self):
        """ Overwrite the 'run' method of threading.Thread
        """
        
        batch_size = self.batch_size
        patch_len = self.patch_len
        feat_dim = self.feat_dim
        num_track = self.num_track
        data_flow = self.data_flow
        
        while self.running:
            if self.data_buffer is None:
                if self.index_start + batch_size <= len(data_flow):
                    # This case means we are still in this epoch
                    batch_index = data_flow[self.index_start : self.index_start + batch_size]
                    self.index_start += batch_size

                elif self.index_start < len(data_flow):
                    # This case means we've come to the
                    # end of this epoch, take all the rest data
                    # and shuffle the training data again
                    batch_index = data_flow[self.index_start:]
                    print "finish a epoch"

                    # Now, we've finished this epoch
                    # let's shuffle it again.
                    self.rng.shuffle(data_flow)
                    self.index_start = 0
                else:
                    # This case means index_start == len(shuffle_index)
                    # Thus, we've finished this epoch
                    # let's shuffle it again.
                    self.rng.shuffle(data_flow)
                    batch_index = data_flow[0: batch_size]
                    self.index_start = batch_size
                    print "finish a epoch exactly"

                # batch_index is a list with length 'batch_size'
                # batch_index[i] = [index_start, index_end]
                # data  : batch_size x T x d
                # label : batch_size x T x d x 2

                data = np.zeros(shape=(batch_size, patch_len, feat_dim))
                label = np.zeros(shape=(batch_size, patch_len, feat_dim, num_track))

                patch_index = 0
                for index_start, index_end in batch_index:
                    data_tmp = (self.f['X'][index_start: index_end, ...] - self.data_mean) / self.data_std
                    label_tmp = self.f['Y'][index_start: index_end, ...]
                    cur_T = index_end - index_start # this should be = patch_len
                    
                    data[patch_index, :cur_T, :] = data_tmp[None, ...]
                    label[patch_index, :cur_T, :, :] = label_tmp[None, ...]

                    patch_index += 1


                with self.lock:
                    self.data_buffer = data, label
            sleep(0.0001)

    def iterate_batch(self):
        while self.data_buffer is None:
            sleep(0.0001)

        data, label = self.data_buffer
        data = np.asarray(data, dtype=np.float32)
        label = np.asarray(label, dtype=np.float32)
        with self.lock:
            self.data_buffer = None

        return data, label

    def close(self):
        
        self.running = False
        self.join()
        self.f.close()





#%%
if __name__ == '__main__':
    """ Test the speed of reader
    """
    config = Config("../config.json")
    file_data = "../data/h5/train/vn-cl.h5"
    file_info = "../data/h5/train/vn-cl.cpickle"
    data_reader = Reader(file_data, file_info, config)
    #%%
    for i in range(5):
        #%%
        t_start = time.time()
        data, label = data_reader.iterate_batch()
        t_end = time.time()
        print('%d-th batch, %.4f seconds ...' % (i + 1, t_end - t_start))

        print data.shape
        print label.shape
#%%
    data_reader.close()

