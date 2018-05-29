from my_reader import Reader
from my_model import MyModel

import time
import logging


#from progress.bar import Bar

import tensorflow as tf
from keras.callbacks import TensorBoard
from util import Config
import numpy as np
import os


#def write_log(callback, names, logs, batch_no):
#    """ Function to generate the tensorboard display.
#    """
#    for name, value in zip(names, logs):
#        summary = tf.Summary()
#        summary_value = summary.value.add()
#        summary_value.simple_value = value
#        summary_value.tag = name
#        callback.writer.add_summary(summary, batch_no)
#        callback.writer.flush()


class Timer:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.total_time = 0
        self.avg_time = 0
        self.n_toc = 0

    def tic(self):
        self.n_toc = 0
        self.start_time = time.time()

    def toc(self):
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time
        self.n_toc += 1.
        self.avg_time = self.total_time / self.n_toc
        return self.total_time


#class Logger:
#    """
#    When receiving a message, first print it on screen, then write it into log file.
#    If save_dir is None, it writes no log and only prints on screen.
#    """
#    def __init__(self, save_dir):
#        if save_dir is not None:
#            self.logger = logging.getLogger()
#            self.logger.setLevel(logging.INFO)
#
#            self.handler = logging.FileHandler(osp.join(save_dir, 'experiment.log'))
#
#            self.handler.setLevel(logging.INFO)
#            formatter = logging.Formatter(fmt='%(asctime)s |  %(message)s')
#            self.handler.setFormatter(fmt=formatter)
#
#            self.logger.addHandler(self.handler)
#
#        else:
#            self.logger = None
#
#    def info(self, msg):
#        print msg
#        if self.logger is not None:
#            self.logger.info(msg)
#
#    def stop(self):
#        self.logger.removeHandler(self.handler)
#        del self.handler
#        del self.logger




class Trainer(object):
    
    def __init__(self, config=None, logger=None, instr_mix=None):
        
        self.config = config
        self.logger = logger
        
        self.batch_size_train = config.get("batch_size_train")
        self.dim_feat = config.get("dim_feat")
        self.batch_size_eval = config.get("batch_size_eval")
        self.num_epoch = config.get("num_epoch")
        self.num_patience = config.get("num_patience")
        path_feat = config.get("path_feat")
        
        self.path_model = config.get("path_model")
        if not os.path.exists(self.path_model):
            os.makedirs(self.path_model)
            
        file_data = os.path.join(path_feat, "train", instr_mix) + ".h5"
        file_info = os.path.join(path_feat, "train", instr_mix) + ".cpickle"
        logger.log("Reading the train data")
        self.data_train = Reader(file_data=file_data, file_info=file_info, config=config)
        
        file_data = os.path.join(path_feat, "valid", instr_mix) + ".h5"
        file_info = os.path.join(path_feat, "valid", instr_mix) + ".cpickle"
        logger.log("Reading the valid data")
        self.data_valid = Reader(file_data=file_data, file_info=file_info, config=config)
        
        file_data = os.path.join(path_feat, "test", instr_mix) + ".h5"
        file_info = os.path.join(path_feat, "test", instr_mix) + ".cpickle"
        logger.log("Reading the test data")
        self.data_test = Reader(file_data=file_data, file_info=file_info, config=config)
        
        
    def eval_loss(self, model, data):
        """ Evaluate the model on the validation or testing sets.
        """
        batch_size_eval = self.batch_size_eval
        dim_feat = self.dim_feat
        loss = []

        batches_per_reader = len(data.data_flow) / batch_size_eval

    #    bar = Bar('Processing', max=batches_per_reader)
        for batch_index in range(batches_per_reader):
    
            x, t = data.iterate_batch()
            loss_eval = model.test_on_batch(x=[x], y=[t])
            loss.append(np.sqrt(loss_eval / ((x.shape[1] * dim_feat) ** 2)))
    #        bar.next()
    #    bar.finish()
        # Reset for next evaluation
        data.reset()
    
        return np.mean(loss)
    
    
    def run(self):
        
        config = self.config
        logger = self.logger
        
        path_model = self.path_model
        
        batch_size_train = self.batch_size_train
        dim_feat = self.dim_feat
        num_epoch = self.num_epoch
        num_patience = self.num_patience
    
        data_train = self.data_train 
        data_valid = self.data_valid
    
        """ 2. Audio model
        """
#        logger.info('Building the model ...')
        logger.log("Building the model")
        
        m = MyModel(config)
        m.build_model()
        
        
        logger.log('set parameters for training ...')
        iters_per_epoch = (len(data_train.data_flow) // batch_size_train) + 1
        maxIters = num_epoch * iters_per_epoch
        logger.log('Train for %d-epochs, %d-iters ...' % (num_epoch, maxIters))
    
        # Initialize the logger and timer
#        logger = Logger(save_dir=path_model)
        timer = Timer()
    
    #    tf_board_path = osp.join(model_path, 'graph') 05/03/2018
    #    train_names = ['train_loss'] 05/03/2018
    #    valid_names = ['valid_loss'] 05/03/2018
    
        

        
        validFreq = iters_per_epoch // 5  # validation about once an epoch
        # validFreq = 2
        dispFreq = iters_per_epoch // 25  # display 5 times each epoch
        # dispFreq = 1
        loss_valid_best = float('inf')
    
        # Optimization
        loss_train_hist = []
        loss_train_hist_avg = []
        loss_valid_hist = []
    
        start_time = time.time()
        iters = 1
    
        best_iter = 0  # The iteration that the best model occurs
    #    callback = TensorBoard(tf_board_path) 05/03/2018
    #    callback.set_model(audio_model) 05/03/2018
        try:
            
            while iters < maxIters:
    
                x_train, t_train = data_train.iterate_batch()
                
                timer.tic()
                loss = m.my_model.train_on_batch(x=[x_train], y=[t_train])
                loss = np.sqrt(loss / ((x_train.shape[1] * dim_feat) ** 2))
    
                loss_train_hist.append(loss)
                loss_train_hist_avg.append(np.mean(loss_train_hist))
    

                """ Write to tensorboard
                """
                # 05/03/2018
    #            tf_board_train = []
    #            tf_board_train.append(loss_train_hist_avg[-1])
    #            write_log(callback, train_names, tf_board_train, iters)
    
                if np.mod(iters, dispFreq) == 0:
                    logger.log('iter={}, training loss = {:.5f}, finish -> {}, time={:.1f} sec'
                                .format(iters, loss_train_hist_avg[-1], maxIters, timer.toc()))
                    
                if np.mod(iters, validFreq) == 0:
                    logger.log('... Computing validation err')
                    loss_valid = self.eval_loss(model=m.my_model, data=data_valid)
    
                    loss_valid_hist.append(loss_valid)
    
                    logger.log('--- this valid loss = {:.4f}, best = {:.4f}'.format(loss_valid, loss_valid_best))
#                    file_model = os.path.join(path_model, 'model.iter') + str(iters)
#                    logger.log('Saving model at iter={}'.format(iters))
#                    m.save_model(filename=file_model)
    
                    # Write to tensorboard
                    # 05/03/2018
    #                tf_board_valid = []
    #                tf_board_valid.append(loss_valid)
    #                write_log(callback, valid_names, tf_board_valid, iters // validFreq)
    
                    # save the model
                    if loss_valid < loss_valid_best:
                        loss_valid_best = loss_valid
                        maxIters = max(maxIters, iters + num_patience * iters_per_epoch)
    
                        file_model = os.path.join(path_model, 'model.best')
                        logger.log('Saving best model at iter={}'.format(iters))
                        m.save_model(filename=file_model)
    
                iters += 1
        except KeyboardInterrupt:
            logger.log('Training interrupted ...')
        end_time = time.time()
    
        # Close the valid iterators
        data_valid.close()
    
        # save the train costs
        file_train_cost = os.path.join(path_model, 'loss.npz')
        np.savez(file_train_cost, loss_train_hist=loss_train_hist,
                 loss_train_hist_avg=loss_train_hist_avg,
                 loss_valid_hist=loss_valid_hist,
                 time=(end_time - start_time) / 60.,
                 data_mean=data_train.data_mean,  # mean of the training data
                 data_std=data_train.data_std,
                 best_iter=best_iter,
                 validFreq=validFreq)  # std of the training data
    
        data_train.close()
    
        # Test
        data_test = self.data_test
        
        file_weights = os.path.join(path_model, 'model.best.h5')
        m.my_model.load_weights(file_weights)
        test_loss = self.eval_loss(model=m.my_model, data=data_test)
        data_test.close()
        logger.log('--- Test loss = {:.4f}'.format(test_loss))
        logger.log('Training done ...')
    
        # Close the reader
        logger.log('Shutting down the reader ...')
        logger.log('Training took {:.2f} minutes in total ...'.format((end_time - start_time) / 60.))
#        logger.stop()

#%%
        
        
from util.logger import Logger
if __name__ == '__main__':
    
    
    config = Config("../config.json")
    logger = Logger(config)
    config.set_logger(logger)
    
    t = Trainer(config, logger)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    