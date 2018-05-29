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

#config = Config("config.json")
#AUD_INPUT_DIM = config.get("AUD_INPUT_DIM")
#NUM_SPK = config.get("NUM_SPK")
#BATCH_SIZE_TRAIN = config.get("BATCH_SIZE_TRAIN")
#BATCH_SIZE_EVAL = config.get("BATCH_SIZE_EVAL")
#BATCH_SIZE_TEST = config.get("BATCH_SIZE_TEST")
#EMBEDDINGS_DIMENSION = config.get("EMBEDDINGS_DIMENSION")



def save_model(model, filename):
    """ Save the trained model
    :param model    : the trained keras model
    :param filename : sth like /.../.../model.best
    :return:
    """
    # serialize model to JSON
    model_json = model.to_json()
    if osp.isfile(filename + '.json'):
        os.system('rm ' + filename + '.json')
    with open(filename + ".json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    if osp.isfile(filename + '.h5'):
        os.system('rm ' + filename + '.h5')
    model.save_weights(filename + ".h5")
    print("Model saved to disk")


def load_model(filename):
    # load json and create model
    json_file = open(filename + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(filename + ".h5")
    print("Model loaded from disk")
    return loaded_model


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


def build_model(path_weights=None, config=None):
    """ Build the dpc model
    """
    
    num_layer = config.get("num_layer")
    dim_hid = config.get("dim_hid")
    dim_feat = config.get("dim_feat")
    dim_embedding = config.get("dim_embedding")
    
    
    
    def l2_normalize(inputs):
        return tf.nn.l2_normalize(inputs, -1)

    # audio_input : batch_size x T_audio x input_size
    audio_input = Input(shape=(None, dim_feat))

    x = audio_input

    """ Recurrent layer
        Add dropout to the output of each LSTM layer except the last one
    """
    for i in range(num_layer):
        # Different forward dropout for each time stamp
        # x = Dropout(rate=ddrop)(x)
        # Same recurrent dropout for each time stamp,
        # 4 gates share the same dropout rate
        x = Bidirectional(LSTM(units=dim_hid, return_sequences=True,
                               implementation=2,
                               recurrent_dropout=0.2))(x)

    # batch_size x T_audio x (input_size x embed_dim)
    audio_embed = TimeDistributed(Dense(dim_feat * dim_embedding, activation='tanh'),
                                  name='audio_embed')(x)
    # batch_size x T_audio x input_size x embed_dim
    audio_embed = KL.Reshape(target_shape=(-1, dim_feat, dim_embedding))(audio_embed)

    # Reshape: batch_size x (T_audio x input_size) x embed_dim
    # And normalize along the last dimension
    audio_embed = KL.Reshape(target_shape=(-1, dim_embedding))(audio_embed)
    audio_embed = KL.Lambda(l2_normalize)(audio_embed)

    # Reshape back to size : batch_size x T_audio x input_size x embed_dim
    audio_embed = KL.Reshape(target_shape=(-1, dim_feat, dim_embedding))(audio_embed)

    model = Model(inputs=[audio_input], outputs=[audio_embed])

    if path_weights:
        model.load_weights(path_weights)

    model.compile(loss=affinity_kmeans, optimizer=Adam(lr=0.001))

    return model
