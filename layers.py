"""
change log:
- Version 1: change the out_grads of `backward` function of `ReLU` layer into inputs_grads instead of in_grads
"""

import numpy as np 
from utils.tools import *

class Layer(object):
    """
    
    """
    def __init__(self, name):
        """Initialization"""
        self.name = name
        self.training = True  # The phrase, if for training then true
        self.trainable = False # Whether there are parameters in this layer that can be trained

    def forward(self, inputs):
        """Forward pass, reture outputs"""
        raise NotImplementedError

    def backward(self, in_grads, inputs):
        """Backward pass, return gradients to inputs"""
        raise NotImplementedError

    def update(self, optimizer):
        """Update parameters in this layer"""
        pass

    def set_mode(self, training):
        """Set the phrase/mode into training (True) or tesing (False)"""
        self.training = training

    def set_trainable(self, trainable):
        """Set the layer can be trainable (True) or not (False)"""
        self.trainable = trainable

    def get_params(self, prefix):
        """Reture parameters and gradients of this layer"""
        return None


class FCLayer(Layer):
    def __init__(self, in_features, out_features, name='fclayer', initializer=Guassian()):
        """Initialization

        # Arguments
            in_features: int, the number of inputs features
            out_features: int, the numbet of required outputs features
            initializer: Initializer class, to initialize weights
        """
        super(FCLayer, self).__init__(name=name)
        self.trainable = True
        self.weights = initializer.initialize((in_features, out_features))
        self.bias = np.zeros(out_features)

        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_features)

        # Returns
            outputs: numpy array with shape (batch, out_features)
        """
        outputs = None
        #############################################################
        outputs = np.dot(inputs, self.weights) + self.bias 
        #############################################################
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass, store gradients to self.weights into self.w_grad and store gradients to self.bias into self.b_grad

        # Arguments
            in_grads: numpy array with shape (batch, out_features), gradients to outputs
            inputs: numpy array with shape (batch, in_features), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_features), gradients to inputs
        """
        out_grads = None
        #############################################################
        out_grads = np.dot(in_grads, self.weights.T)
        self.w_grad = np.dot(inputs.T, in_grads)
        self.b_grad = np.dot(in_grads.T, np.ones(inputs.shape[0]))

        #############################################################
        return out_grads

    def update(self, params):
        """Update parameters (self.weights and self.bias) with new params
        
        # Arguments
            params: dictionary, one key contains 'weights' and the other contains 'bias'

        # Returns
            none
        """
        for k,v in params.items():
            if 'weights' in k:
                self.weights = v
            else:
                self.bias = v
        
    def get_params(self, prefix):
        """Return parameters (self.weights and self.bias) as well as gradients (self.w_grad and self.b_grad)
        
        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer, one key contains 'weights' and the other contains 'bias'
            grads: dictionary, store gradients of this layer, one key contains 'weights' and the other contains 'bias'

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix+':'+self.name+'/weights': self.weights,
                prefix+':'+self.name+'/bias': self.bias
            }
            grads = {
                prefix+':'+self.name+'/weights': self.w_grad,
                prefix+':'+self.name+'/bias': self.b_grad
            }
            return params, grads
        else:
            return None

class Convolution(Layer):
    def __init__(self, conv_params, initializer=Guassian(), name='conv'):
        """Initialization

        # Arguments
            conv_params: dictionary, containing these parameters:
                'kernel_h': The height of kernel.
                'kernel_w': The width of kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels padded to the bottom, top, left and right of each feature map. Here, pad=2 means a 2-pixel border of padded with zeros.
                'in_channel': The number of input channels.
                'out_channel': The number of output channels.
            initializer: Initializer class, to initialize weights
        """
        super(Convolution, self).__init__(name=name)
        self.trainable = True
        self.kernel_h = conv_params['kernel_h'] # height of kernel
        self.kernel_w = conv_params['kernel_w'] # width of kernel
        self.pad = conv_params['pad']
        self.stride = conv_params['stride']
        self.in_channel = conv_params['in_channel']
        self.out_channel = conv_params['out_channel']

        self.weights = initializer.initialize((self.out_channel, self.in_channel, self.kernel_h, self.kernel_w))
        self.bias = np.zeros((self.out_channel))

        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, out_channel, out_height, out_width)
        """
        outputs = None
        #############################################################
        strides = self.stride
        pads = self.pad
        (N, C, H, W) = inputs.shape
        (kernel_N, kernel_C, kernel_H, kernel_W) = self.weights.shape
        # calculating the height and width of next layer using formula
        out_height = 1 + int((H + 2 * pads - kernel_H) / strides)
        out_width = 1 + int((W + 2 * pads - kernel_W) / strides)

        inputs_pad = np.pad(inputs, ((0, 0), (0, 0), (pads, pads), (pads, pads)), 'constant', constant_values=0)
        
        i0 = np.repeat(np.arange(kernel_H), kernel_W)
        i0 = np.tile(i0, C)
        i1 = strides * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(kernel_W), kernel_H * C)
        j1 = strides *np.tile(np.arange(out_width), out_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        k = np.repeat(np.arange(C), kernel_H * kernel_W).reshape(-1, 1)

        cols = inputs_pad[:, k, i , j]
        X_col = cols.transpose(1, 2, 0).reshape(kernel_H * kernel_W * C, -1)
        W_col = self.weights.reshape(kernel_N, -1)

        out = W_col @ X_col + self.bias.reshape((kernel_N,1))
        out = out.reshape(kernel_N, out_height,  out_width, N)
        outputs = out.transpose(3, 0, 1, 2)
        #############################################################
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass, store gradients to self.weights into self.w_grad and store gradients to self.bias into self.b_grad

        # Arguments
            in_grads: numpy array with shape (batch, out_channel, out_height, out_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs
        """
        out_grads = None
        #############################################################
        strides = self.stride
        pads = self.pad
        (N, C, H, W) = inputs.shape
        (d_N, d_C, d_H, d_W) = in_grads.shape
        kernel_h = self.kernel_h
        kernel_w = self.kernel_w
        W = self.weights
        dinputs = np.zeros(inputs.shape)  
        inputs_pad = np.pad(inputs, ((0, 0), (0, 0), (pads, pads), (pads, pads)), 'constant', constant_values=0)
        dinputs_pad = np.pad(dinputs, ((0, 0), (0, 0), (pads, pads), (pads, pads)), 'constant', constant_values=0)
        
        for i in range(d_N):
            i_inputs_pad = inputs_pad[i]
            i_dinputs_pad = dinputs_pad[i]
            for c in range(d_C):
                for h in range(d_H):
                    for w in range(d_W):            
                        v_start = h * strides
                        v_end = v_start + kernel_h
                        h_start = w * strides
                        h_end = h_start + kernel_w
                        #print(i_dinputs_pad[:, v_start:v_end, h_start:h_end].shape)
                        i_dinputs_pad[:, v_start:v_end, h_start:h_end] += W[c,:,:,:] * in_grads[i,c,h,w]
                        self.w_grad[c,:,:,:] += i_inputs_pad[:, v_start:v_end, h_start:h_end] * in_grads[i, c, h, w]
                        self.b_grad[c] += in_grads[i, c, h, w] 
                        
                        # Update gradients for the window and the filter's parameters using the code formulas given above
                        #da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                        #dW[:,:,:,c] += i_inputs_pad[:, v_start:v_end, h_start:h_end] * dZ[i, h, w, c]
                        #db[:,:,:,c] += dZ[i, h, w, c]
        
            if (pads == 0):
                dinputs[i, :, :, :] = i_dinputs_pad[: , :, :]
            else:
                dinputs[i, :, :, :] = i_dinputs_pad[: ,pads:-pads, pads:-pads]
                  
        #############################################################
        return dinputs

    def update(self, params):
        """Update parameters (self.weights and self.bias) with new params
        
        # Arguments
            params: dictionary, one key contains 'weights' and the other contains 'bias'

        # Returns
            none
        """
        for k,v in params.items():
            if 'weights' in k:
                self.weights = v
            else:
                self.bias = v

    def get_params(self, prefix):
        """Return parameters (self.weights and self.bias) as well as gradients (self.w_grad and self.b_grad)
        
        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer, one key contains 'weights' and the other contains 'bias'
            grads: dictionary, store gradients of this layer, one key contains 'weights' and the other contains 'bias'

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix+':'+self.name+'/weights': self.weights,
                prefix+':'+self.name+'/bias': self.bias
            }
            grads = {
                prefix+':'+self.name+'/weights': self.w_grad,
                prefix+':'+self.name+'/bias': self.b_grad
            }
            return params, grads
        else:
            return None

class ReLU(Layer):
    def __init__(self, name='relu'):
        """Initialization
        """
        super(ReLU, self).__init__(name=name)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array

        # Returns
            outputs: numpy array
        """
        outputs = np.maximum(0, inputs)
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array, gradients to outputs
            inputs: numpy array, same with forward inputs

        # Returns
            out_grads: numpy array, gradients to inputs 
        """
        inputs_grads = (inputs >=0 ) * in_grads
        out_grads = inputs_grads
        return out_grads


# TODO: add padding
class Pooling(Layer):
    def __init__(self, pool_params, name='pooling'):
        """Initialization

        # Arguments
            pool_params is a dictionary, containing these parameters:
                'pool_type': The type of pooling, 'max' or 'avg'
                'pool_h': The height of pooling kernel.
                'pool_w': The width of pooling kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels that will be used to zero-pad the input in each x-y direction. Here, pad=2 means a 2-pixel border of padding with zeros.
        """
        super(Pooling, self).__init__(name=name)
        self.pool_type = pool_params['pool_type']
        self.pool_height = pool_params['pool_height']
        self.pool_width = pool_params['pool_width']
        self.stride = pool_params['stride']
        self.pad = pool_params['pad']

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, in_channel, out_height, out_width)
        """
        outputs = None
        #############################################################
        strides = self.stride
        pads = self.pad
        (N, C, H, W) = inputs.shape
        inputs_reshaped = inputs.reshape(N * C, 1, H, W)
        (N_P, N_C, N_H, N_W) = inputs_reshaped.shape
        pool_H = self.pool_height
        pool_W = self.pool_width
        # calculating the height and width of next layer using formula
        inputs_reshaped_pad = np.pad(inputs_reshaped, ((0, 0), (0, 0), (pads, pads), (pads, pads)), 'constant', constant_values=0)

        out_height = 1 + int((H + 2 * pads - pool_H) / strides)
        out_width = 1 + int((W + 2 * pads - pool_W) / strides)

        i0 = np.repeat(np.arange(pool_H), pool_W)
        i0 = np.tile(i0, N_C)
        i1 = strides * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(pool_W), pool_H * N_C)
        j1 = strides *np.tile(np.arange(out_width), out_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        k = np.repeat(np.arange(N_C), pool_H * pool_W).reshape(-1, 1)

        cols = inputs_reshaped_pad[:, k, i , j]
        self.X_col = cols.transpose(1, 2, 0).reshape(pool_H * pool_W * N_C, -1)

        if self.pool_type =='max':
            self.max_ind = np.argmax(self.X_col, axis=0)
            outputs = self.X_col[self.max_ind, range(self.max_ind.size)]
        elif self.pool_type == 'avg':
            self.avg = np.mean(self.X_col, axis=0)
            outputs = self.avg

        outputs = outputs.reshape(out_height, out_width, N, C)
        outputs = outputs.transpose(2, 3, 0, 1)
        #############################################################
        
        return outputs
        
    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array with shape (batch, in_channel, out_height, out_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs
        """
        out_grads = None
        #############################################################
        strides = self.stride
        pads = self.pad
        pool_H = self.pool_height
        pool_W = self.pool_width
        # calculating the height and width of next layer using formula
        (N, d_C, d_H, d_W) = inputs.shape
        (N, C, H, W) = in_grads.shape
        
        out_grads = np.zeros(inputs.shape)
        
        # looping through all N examples
        for i in range(N):
            i_inputs = inputs[i,:,:,:]
            # looping through the height, width and channels 
            for h in range(H):
                for w in range(W):
                    for c in range(C):
                        v_start = h * strides
                        v_end = v_start + pool_H
                        h_start = w * strides
                        h_end = h_start + pool_W    

                        if self.pool_type == 'max':
                            i_out = i_inputs[c, v_start:v_end, h_start:h_end]
                            mask = (i_out == np.max(i_out))
                            out_grads[i, c, v_start:v_end, h_start:h_end] += mask * in_grads[i, c, h, w]
                        elif self.pool_type == 'avg':
                            in_grads2 = in_grads[i, c, h, w]
                            average = in_grads2 / (pool_H * pool_W)
                            out_grads[i, c, v_start:v_end, h_start:h_end] += np.ones((pool_H, pool_W)) * average
        #############################################################
        assert(out_grads.shape == inputs.shape)

        return out_grads



class Dropout(Layer):
    def __init__(self, ratio, name='dropout', seed=None):
        """Initialization

        # Arguments
            ratio: float [0, 1], the probability of setting a neuron to zero
            seed: int, random seed to sample from inputs, so as to get mask. (default as None)
        """
        super(Dropout, self).__init__(name=name)
        self.ratio = ratio
        self.mask = None
        self.seed = seed

    def forward(self, inputs):
        """Forward pass (Hint: use self.training to decide the phrase/mode of the model)

        # Arguments
            inputs: numpy array

        # Returns
            outputs: numpy array
        """
        outputs = inputs
        np.random.seed(self.seed)
        #############################################################
        if self.training:
            if self.mask is None:
                self.mask = np.random.binomial(1, (1-self.ratio), size=inputs.shape)
            outputs *= self.mask
            outputs = inputs * (1/(1-self.ratio))
            #self.mask = (np.random.rand(*inputs.shape) < (1-self.ratio)) / (1-self.ratio)
            #outputs = (inputs * self.mask) 
        #############################################################
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array, gradients to outputs
            inputs: numpy array, same with forward inputs

        # Returns
            out_grads: numpy array, gradients to inputs 
        """
        out_grads = in_grads
        #############################################################
        if self.training:
            out_grads *= self.mask * (1 / (1-self.ratio))
      
        #############################################################

        return out_grads

class Flatten(Layer):
    def __init__(self, name='flatten', seed=None):
        """Initialization
        """
        super(Flatten, self).__init__(name=name)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, in_channel*in_height*in_width)
        """
        batch = inputs.shape[0]
        outputs = inputs.copy().reshape(batch, -1)
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array with shape (batch, in_channel*in_height*in_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs 
        """
        out_grads = in_grads.copy().reshape(inputs.shape)
        return out_grads
        
