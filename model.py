# %% 搭建Res2Net+SE+attentive statistic pooling
import numpy as np
from tensorflow.keras.layers import Layer, Conv1D, Conv2D, Activation, Input, multiply, Flatten, GlobalMaxPooling1D, \
    Dropout
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers import Reshape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import regularizers, constraints, initializers

class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.conv1a = tf.keras.layers.Conv1D(64, 3)
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, dilation=1, training=False):
        filter_num = input_tensor.shape.dims[-1]
        per_num = filter_num // 4
        ls_input = BatchNormalization(momentum=0.99)(input_tensor)
        ls_input = Activation('relu')(ls_input)
        input_tensor = Conv1D(filter_num, 1, dilation_rate=dilation)(input_tensor)
        input_tensor = BatchNormalization(momentum=0.99)(input_tensor)
        input_tensor = Activation('relu')(input_tensor)
        ans1 = input_tensor[:, :, 0:per_num]

        data2 = input_tensor[:, :, per_num:2 * per_num]
        ans2 = Conv1D(per_num, 5, dilation_rate=dilation, padding='same')(data2)
        ans2 = BatchNormalization(momentum=0.99)(ans2)
        ans2 = Activation('relu')(ans2)

        data3 = input_tensor[:, :, 2 * per_num:3 * per_num]
        data3 += ans2
        data3 = BatchNormalization(momentum=0.99)(data3)
        data3 = Activation('relu')(data3)
        ans3 = Conv1D(per_num, 5, dilation_rate=dilation, padding='same')(data3)
        ans3 = BatchNormalization(momentum=0.99)(ans3)
        ans3 = Activation('relu')(ans3)

        data4 = input_tensor[:, :, 3 * per_num:]
        data4 += ans3
        data4 = BatchNormalization(momentum=0.99)(data4)
        data4 = Activation('relu')(data4)
        ans4 = Conv1D(per_num, 5, dilation_rate=dilation, padding='same')(data4)
        ans4 = BatchNormalization(momentum=0.99)(ans4)
        ans4 = Activation('relu')(ans4)

        ans = tf.concat([ans1, ans2, ans3, ans4], 2)
        ans = BatchNormalization(momentum=0.99)(ans)
        ans = Activation('relu')(ans)

        fin_ans = Conv1D(filter_num, 1, dilation_rate=dilation, padding='same')(ans)
        fin_ans += ls_input
        fin_ans = BatchNormalization(momentum=0.99)(fin_ans)
        fin_ans = Activation('relu')(fin_ans)
        return fin_ans

class Attention_layer(Layer):

    def __init__(self, want_pooling=True,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 V_regularizer=None, k_regularizer=None,
                 V_constraint=None, k_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.want_pooling=want_pooling
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.V_regularizer = regularizers.get(V_regularizer)
        self.k_regularizer = regularizers.get(k_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.V_constraint = constraints.get(V_constraint)
        self.k_constraint = constraints.get(k_constraint)

        self.bias = bias
        super(Attention_layer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'supports_masking': self.supports_masking,
            'init': self.init,
            'want_pooling': self.want_pooling,
            'W_regularizer': self.W_regularizer,
            'b_regularizer': self.b_regularizer,
            'V_regularizer': self.V_regularizer,
            'k_regularizer': self.k_regularizer,
            'W_constraint': self.W_constraint,
            'b_constraint': self.b_constraint,
            'V_constraint': self.V_constraint,
            'k_constraint': self.k_constraint,
            'bias': self.bias
        })
        return config

    def build(self, input_shape):
        # assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],), initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        # V也可以有两个维度，这样就会使得对于input中的每一个点都得到自己的注意力分数，值得尝试
        self.V = self.add_weight(shape=(input_shape[-1], 1), initializer=self.init,
                                 name='{}_V'.format(self.name),
                                 regularizer=self.V_regularizer,
                                 constraint=self.V_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-2], 1),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
            self.k = self.add_weight(shape=(input_shape[-2], 1),
                                     initializer='zero',
                                     name='{}_k'.format(self.name),
                                     regularizer=self.k_regularizer,
                                     constraint=self.k_constraint)

        super(Attention_layer, self).build(input_shape)

    # def compute_mask(self, input, input_mask=None):
    #     # do not pass the mask to the next layers
    #     return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W)
        # uit = tf.transpose(uit,perm=[1,0,2]) # 交换维度
        if self.bias:
            uit += self.b
        uit = K.dot(uit, self.V)
        # uit = tf.transpose(uit,perm=[1,0,2]) # 交换维度
        if self.bias:
            uit += self.k
        uit = K.tanh(uit)
        a = K.exp(uit)
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        # print(a)
        # a=tf.squeeze(a,0)
        # a = K.expand_dims(a)
        # print(x)
        weighted_input = x * a
        if self.want_pooling == False:
            return weighted_input
        # print(weighted_input)
        mean = tf.reduce_sum(weighted_input, 1)
        h_2 = tf.multiply(x, x)
        a_h_2 = a * h_2
        sum_a_h_2 = tf.reduce_sum(a_h_2, 1)
        std_2 = sum_a_h_2 - tf.multiply(mean, mean)
        out = tf.concat([mean, std_2], 1)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def se_res2block(x, dilation, filter, strides=5):
    block = ResnetIdentityBlock(1, [1, 2, 3])
    out = Conv1D(filter, strides, padding='same', dilation_rate=dilation)(x)
    out = BatchNormalization(momentum=0.99)(out)
    out = Activation('relu')(out)
    squeeze = GlobalMaxPooling1D()(out)
    excitation = Dense(filter * 2)(squeeze)
    excitation = Activation('tanh')(excitation)
    excitation = Dense(filter)(excitation)
    excitation = Activation('relu')(excitation)
    excitation = Reshape((1, filter))(excitation)

    out = multiply([out, excitation])

    out = block.call(out, dilation=dilation)

    squeeze = GlobalMaxPooling1D()(out)
    excitation = Dense(filter * 2)(squeeze)
    excitation = Activation('tanh')(excitation)
    excitation = Dense(filter)(excitation)
    excitation = Activation('relu')(excitation)
    excitation = Reshape((1, filter))(excitation)

    out = multiply([out, excitation])
    out = Conv1D(filter, strides, padding='same')(out)
    out = BatchNormalization(momentum=0.99)(out)
    return Activation('relu')(out)

def se_block(x):
    filter = x.shape.dims[-1]
    squeeze = GlobalMaxPooling1D()(x)
    excitation = Dense(2 * filter)(squeeze)
    excitation = Activation('tanh')(excitation)
    excitation = Dense(filter)(excitation)
    excitation = Activation('relu')(excitation)
    excitation = Reshape((1, filter))(excitation)

    return multiply([x, excitation])

def fc(x, out_node_num, dropout=0.3, momentum=0.99, activation_fun='relu'):
    out = Dropout(dropout)(x)
    out = Dense(out_node_num)(out)
    out = BatchNormalization(momentum=momentum)(out)
    return Activation(activation_fun)(out)

# x_train=np.array([x.reshape((128,259)) for x in x_train])
# x_train = tf.transpose(x_train,perm=[0,2,1]) # 交换维度
# x_test=np.array([x.reshape((128,259)) for x in x_test])
# x_test = tf.transpose(x_test,perm=[0,2,1]) # 交换维度
inp = Input(shape=(200, 24))
out1 = se_res2block(inp, filter=24, strides=3, dilation=2)
out2 = se_res2block(out1, filter=24, strides=3, dilation=3)
out3 = se_res2block(out2, filter=24, strides=3, dilation=4)
out = tf.concat([out1, out2, out3], 2)
out = se_block(out)
# out = attentive_statistic_pooling(out,128)
out = Attention_layer(want_pooling=False)(out)
# a=Attention_layer()
# a.build(input_shape=(188,72))
# a.call(out)
out = Conv1D(64,5,strides=3)(out)
out = BatchNormalization(momentum=0.99)(out)
out = Activation('relu')(out)

out = Conv1D(16,3,strides=2)(out)
out = BatchNormalization(momentum=0.99)(out)
out = Activation('relu')(out)

out = Flatten()(out)
out = BatchNormalization(momentum=0.99)(out)

# out = fc(out,1024,dropout=0.2,activation_fun='relu')

out = fc(out, 64, dropout=0.2, activation_fun='relu')

predictions = Dense(3, activation="softmax")(out)

model = Model(inp, predictions)
model.summary()