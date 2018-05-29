from keras import backend as K
import keras.layers as KL
from keras.models import Model
from keras.layers import Input, Dense, LSTM, multiply
from keras.layers import TimeDistributed, Bidirectional
from keras.optimizers import Adam
from keras.models import model_from_json
import tensorflow as tf
import os.path as osp
import os
#%%
class MyModel(object):
    
    def __init__(self, config=None):
        
        self.num_track = config.get("num_track")
        self.feat_dim = config.get("dim_feat")
        self.num_layer = config.get("num_layer")
        self.hid_dim = config.get("dim_hid")
        self.dim_embedding = config.get("dim_embedding")

    def build_model(self, path_weights=None):
        """ Build the dpc model
        """
        num_layer = self.num_layer
        hid_dim = self.hid_dim
        dim_input = self.feat_dim
        dim_embedding = self.dim_embedding
        
        def l2_normalize1(inputs):
            return tf.nn.l2_normalize(inputs, -1)
    
        # audio_input : batch_size x T_audio x input_size
        audio_input = Input(shape=(None, dim_input))
    
        x = audio_input
    
        """ Recurrent layer
            Add dropout to the output of each LSTM layer except the last one
        """
        for i in range(num_layer):
            # Different forward dropout for each time stamp
            # x = Dropout(rate=ddrop)(x)
            # Same recurrent dropout for each time stamp,
            # 4 gates share the same dropout rate
            x = Bidirectional(LSTM(units=hid_dim, return_sequences=True,
                                   implementation=2,
                                   recurrent_dropout=0.2))(x)
    
        # batch_size x T_audio x (input_size x embed_dim)
        audio_embed = TimeDistributed(Dense(dim_input * dim_embedding, activation='tanh'),
                                      name='audio_embed')(x)
        # batch_size x T_audio x input_size x embed_dim
        audio_embed = KL.Reshape(target_shape=(-1, dim_input, dim_embedding))(audio_embed)
    
        # Reshape: batch_size x (T_audio x input_size) x embed_dim
        # And normalize along the last dimension
        audio_embed = KL.Reshape(target_shape=(-1, dim_embedding))(audio_embed)
        audio_embed = KL.Lambda(l2_normalize1)(audio_embed)
    
        # Reshape back to size : batch_size x T_audio x input_size x embed_dim
        audio_embed = KL.Reshape(target_shape=(-1, dim_input, dim_embedding))(audio_embed)
    
        my_model = Model(inputs=[audio_input], outputs=[audio_embed])
        # audio_input: [?,?,129]
        # audio_embed: [?,?,129,400]
    
        if path_weights:
            my_model.load_weights(path_weights)
    
        my_model.compile(loss=affinity_kmeans, optimizer=Adam(lr=0.001))
    
        self.my_model = my_model

        
    def save_model(self, filename):
        """ Save the trained model
        :param model    : the trained keras model
        :param filename : sth like /.../.../model.best
        :return:
        """
        my_model = self.my_model
        # serialize model to JSON
        model_json = my_model.to_json()
        if osp.isfile(filename + '.json'):
            os.system('rm ' + filename + '.json')
        with open(filename + ".json", "w") as json_file:
            json_file.write(model_json)
    
        # serialize weights to HDF5
        if osp.isfile(filename + '.h5'):
            os.system('rm ' + filename + '.h5')
        my_model.save_weights(filename + ".h5")
        print("Model saved to disk")


    def load_model(self, filename):
        # load json and create model
        json_file = open(filename + '.json', 'r')
        my_model_json = json_file.read()
        json_file.close()
        my_model = model_from_json(my_model_json)
        # load weights into new model
        my_model.load_weights(filename + ".h5")
        print("Model loaded from disk")
        self.my_model = my_model


def affinity_kmeans(Y, V):
    """ Always put true at the first position
        and pred at the second position
    :param Y : True, with size batch_size x T x d x 2
    :param V : Pred, with size batch_size x T x d x emb
    :return:
    """
    dim_embedding = int(str(V.shape[3]))
    num_track = 2
    def norm(tensor):
        """ frobenius norm
        """
        square_tensor = K.square(tensor)
        frobenius_norm2 = K.sum(square_tensor, axis=(1, 2))
        return frobenius_norm2

    def dot(x, y):
        """ batch dot
        :param x : batch_size x emb1 x (T x d)
        :param y : batch_size x (T x d) x emb2
        """
        return K.batch_dot(x, y, axes=[2, 1])

    def T(x):
        return K.permute_dimensions(x, [0, 2, 1])

    # V : batch_size x (T x d) x emb
    V = KL.Reshape(target_shape=(-1, dim_embedding))(V)

    # Y: batch_size x (T x d) x 2
    Y = KL.Reshape(target_shape=(-1, num_track))(Y)

    # silence_mask : batch_size x (T x d) x 1
    silence_mask = K.sum(Y, axis=2, keepdims=True)
    V = silence_mask * V

    # return with size (batch_size, )
    return norm(dot(T(V), V)) - norm(dot(T(V), Y)) * 2 + norm(dot(T(Y), Y))


