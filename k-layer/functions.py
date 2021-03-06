import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.stats import mode
from copy import deepcopy,copy
import sys

def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)
def relu(x):
    return np.maximum(0,x)

def LoadBatch(filename):
    """ Copied from the dataset website """
    with open('Dataset/'+filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def ComputeCost(X, Y, W , b, lam):
    P = softmax(W @ X + b) # (K,n)
    N = X.shape[1]
    l = -Y*np.log(P) # categorical cross-entropy
    return 1. / N * np.sum(l) + lam * np.sum(W**2)

def ComputeGradsNum(X, Y, P, W, b, lamda, h):
    """ Converted from matlab code """
    no 	= 	W.shape[0]
    d 	= 	X.shape[0]

    grad_W = np.zeros(W.shape);
    grad_b = np.zeros((no, 1));

    c = ComputeCost(X, Y, W, b, lamda);
    
    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] += h
        c2 = ComputeCost(X, Y, W, b_try, lamda)
        grad_b[i] = (c2-c) / h

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i,j] += h
            c2 = ComputeCost(X, Y, W_try, b, lamda)
            grad_W[i,j] = (c2-c) / h

    return grad_W, grad_b

def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h):
    """ Converted from matlab code """
    no 	= 	W.shape[0]
    d 	= 	X.shape[0]

    grad_W = np.zeros(W.shape);
    grad_b = np.zeros((no, 1));
    
    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] -= h
        c1 = ComputeCost(X, Y, W, b_try, lamda)

        b_try = np.array(b)
        b_try[i] += h
        c2 = ComputeCost(X, Y, W, b_try, lamda)

        grad_b[i] = (c2-c1) / (2*h)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i,j] -= h
            c1 = ComputeCost(X, Y, W_try, b, lamda)

            W_try = np.array(W)
            W_try[i,j] += h
            c2 = ComputeCost(X, Y, W_try, b, lamda)

            grad_W[i,j] = (c2-c1) / (2*h)

    return grad_W, grad_b

def montage(W):
    """ Display the image for each label in W """
    fig, ax = plt.subplots(2,5)
    for i in range(2):
        for j in range(5):
            im  = W[i*5+j,:].reshape(32,32,3, order='F')
            sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
            sim = sim.transpose(1,0,2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title("y="+str(5*i+j))
            ax[i][j].axis('off')
    plt.show()

def save_as_mat(data, name="model"):
    """ Used to transfer a python model to matlab """
    sio.savemat(name+'.mat',{name:b})

def load_batch(filename):
    '''
    path = "cifar-10-batches-py/"
    d = LoadBatch(path + filename)
    for key, value in d.items() :
        print (key)
    data = d[b'data']
    labels = d[b'labels']
    batch_labels = d[b'batch_labels']
    '''
    #train_images, train_labels, test_images, test_labels = cifar10()
    #return data,labels,batch_labels


def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_cifar_10_data_onebatch(data_dir, negatives=False):
    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    cifar_train_data_dict = unpickle(data_dir + "/data_batch_1")
    cifar_train_data = cifar_train_data_dict[b'data']
    #else:
    #    cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
    cifar_train_filenames += cifar_train_data_dict[b'filenames']
    cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)

    # validation
    cifar_val_data = None
    cifar_val_filenames = []
    cifar_val_labels = []

    cifar_val_data_dict = unpickle(data_dir + "/data_batch_2")
    cifar_val_data = cifar_val_data_dict[b'data']
    #else:
    #    cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
    cifar_val_filenames += cifar_val_data_dict[b'filenames']
    cifar_val_labels += cifar_val_data_dict[b'labels']

    cifar_val_data = cifar_val_data.reshape((len(cifar_val_data), 3, 32, 32))
    if negatives:
        cifar_val_data = cifar_val_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_val_data = np.rollaxis(cifar_val_data, 1, 4)
    cifar_val_filenames = np.array(cifar_val_filenames)
    cifar_val_labels = np.array(cifar_val_labels)

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    # fix shapes and values of data
    N = 10000
    cifar_train_data = np.reshape(cifar_train_data, (N, 32*32*3))
    cifar_train_data = cifar_train_data / 255.
    cifar_val_data = np.reshape(cifar_val_data, (N, 32*32*3))
    cifar_val_data = cifar_val_data / 255.
    cifar_test_data = np.reshape(cifar_test_data, (N, 32*32*3))
    cifar_test_data = cifar_test_data / 255.

    # one-hot encodings
    train_onehot = np.zeros((N, 10))
    train_onehot[np.arange(N), cifar_train_labels] = 1
    val_onehot = np.zeros((N, 10))
    val_onehot[np.arange(N), cifar_val_labels] = 1
    test_onehot = np.zeros((N, 10))
    test_onehot[np.arange(N), cifar_test_labels] = 1

    return cifar_train_data, cifar_train_filenames, cifar_train_labels, train_onehot.T, \
        cifar_val_data, cifar_val_filenames, cifar_val_labels, val_onehot.T, \
        cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names, test_onehot.T

def load_cifar_10_data(data_dir, N_val, negatives=False):
    """
    Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
    """

    # get the meta_data_dict
    # num_cases_per_batch: 1000
    # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # num_vis: :3072

    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)     


    N_train = cifar_train_data.shape[0] - N_val


    # fix validation and train
    cifar_val_data = cifar_train_data[N_train:]
    cifar_val_filenames = cifar_train_filenames[N_train:]
    cifar_val_labels = cifar_train_labels[N_train:]
    
    cifar_train_data = cifar_train_data[:N_train]
    cifar_train_filenames = cifar_train_filenames[:N_train]
    cifar_train_labels = cifar_train_labels[:N_train]

    # test data
    # cifar_test_data_dict
    # 'batch_label': 'testing batch 1 of 1'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)
    N_test = cifar_test_data.shape[0]

    print("N_train: ", N_train)
    print("N_val: ", N_val)
    print("N_test: ", N_test)

    # fix shapes and values of data
    cifar_train_data = np.reshape(cifar_train_data, (N_train, 32*32*3))
    cifar_train_data = cifar_train_data / 255.
    cifar_val_data = np.reshape(cifar_val_data, (N_val, 32*32*3))
    cifar_val_data = cifar_val_data / 255.
    cifar_test_data = np.reshape(cifar_test_data, (N_test, 32*32*3))
    cifar_test_data = cifar_test_data / 255.

    # one-hot encodings
    train_onehot = np.zeros((N_train, 10))
    train_onehot[np.arange(N_train), cifar_train_labels] = 1
    val_onehot = np.zeros((N_val, 10))
    val_onehot[np.arange(N_val), cifar_val_labels] = 1
    test_onehot = np.zeros((N_test, 10))
    test_onehot[np.arange(N_test), cifar_test_labels] = 1

    return cifar_train_data, cifar_train_filenames, cifar_train_labels, train_onehot.T, \
        cifar_val_data, cifar_val_filenames, cifar_val_labels, val_onehot.T, \
        cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names, test_onehot.T


class Net:
    # d: input dim,
    # m: hidden dim
    # K: out dim
    def __init__(self, dims, batchnorm=False):
        self.params = {}
        self.dims = dims
        self.n_layers = len(dims)-1 
        self.n_hidden = self.n_layers - 1
        self.batchnorm = batchnorm
        if batchnorm:
            self.mus = []
            self.vas = []
        # W: outdim x indim
        # b: outdim x 1
        for l in range(self.n_layers):
            #self.params['W'+str(l)] = self.he_initW(l)
            mu = 0
            sig = 1e-4
            self.params['W'+str(l)] = self.rand_initW(l,mu,sig)
            self.params['b'+str(l)] = self.initb(l)
        if batchnorm:
            for l in range(self.n_layers - 1):
                self.params['gam'+str(l)] = self.initgamma(l)
                self.params['beta'+str(l)] = self.initbeta(l)
                self.mus.append(np.zeros((self.dims[l+1],1)))
                self.vas.append(np.zeros((self.dims[l+1],1)))

    def W(self, i):
        return self.params['W'+str(i)]
    def b(self, i):
        return self.params['b'+str(i)]
    def gam(self, i):
        return self.params['gam'+str(i)]
    def beta(self, i):
        return self.params['beta'+str(i)]

    def he_initW(self, layer):
        return np.random.randn(self.dims[layer+1], self.dims[layer]) * \
                np.sqrt(2./self.dims[layer])

    def rand_initW(self, layer, mean, sig):
        return np.random.normal(
            mean,sig, 
            (self.dims[layer+1], self.dims[layer])
        )

    def initb(self, layer):
        return np.zeros((self.dims[layer+1], 1))

    def initgamma(self, layer):
        return np.ones((self.dims[layer+1], 1))

    def initbeta(self, layer):
        return np.zeros((self.dims[layer+1], 1))

    def print_params(self):
        for param, val in self.params.items():
            print("{} = {}".format(param,val))

    def print_params_shape(self):
        for param, val in self.params.items():
            print("{} : {}".format(param,val.shape))

    # forward k-layer without or with batch normalization
    # foreach layer set means (mus) and variances (vas) to use pre-computed 
    # without batchnorm returns list of X_l, final output P, otherwise:
    # returns list of X_l, final out P, (computed means, computed variances)
    # last tuple is ([],[]) if using pre-computed means and vars
    def forward(self,X_in, training=False, testing=False):
        X_list = [] 
        S_list = []
        Shat_list = []
        X_l = X_in
        N, _ = X_in.shape
        new_mus = []
        new_vas = []
        # all layers except out layer
        for l in range(self.n_layers):
            X_list.append(X_l)
            S_l = self.W(l) @ X_l + self.b(l)

            if l < self.n_hidden:
                S_list.append(S_l)
                dim,_ = S_l.shape
                #print("S_l",S_l)
                # get mean and variance
                if self.batchnorm:
                    if testing:
                        mu = self.mus[l]
                        var = self.vas[l]
                    else:
                        mu = np.mean(S_l,axis=1,keepdims=True)
                        new_mus.append(mu)
                        var = float(N-1)/N \
                                * np.var(S_l, axis=1, keepdims=True) 
                        new_vas.append(var)
                        if training:
                            a = 0.8 # alpha
                            self.mus[l] = a * self.mus[l] + (1-a) * mu 
                            self.vas[l] = a * self.vas[l] + (1-a) * var
                    # batch normalize
                    eps= np.finfo(np.float64).eps
                    vareps = var + eps
                    S_l = 1. / np.sqrt(np.diag(vareps)) * (S_l - mu)
                    Shat_list.append(S_l)
                    # scale
                    S_l = self.gam(l) * S_l + self.beta(l)
                # END batchnorm
                # activation function
                X_l = relu(S_l)
                #X_list.append(X_l)
            else:
                # output layer
                P = softmax(S_l)
        
        #if np.isnan(P).any():
        #    print("X_l ", X_l)
        #    sys.exit()

        if self.batchnorm:
            return S_list, Shat_list, X_list, P, (new_mus, new_vas)
        return X_list, P

    # Y: one-hot, (K, n)
    # X: data are cols
    # lam: penalty term
    # returns cross-entropy loss
    def compute_cost(self,X, Y, lam, testing=False):
        if self.batchnorm:
            _,_,_,P,_ = self.forward(X, testing) # (K,n)
        else:
            _,P = self.forward(X) # (K,n)
        N = X.shape[1]
        crossl = -Y*np.log(P) # categorical cross-entropy
        reg = 0
        for l in range(self.n_layers - 1):
            reg += np.sum(self.W(l)**2)
        loss = 1. / N * np.sum(crossl) + lam * reg
        return loss

    def compute_accuracy(self, X, y, testing=False):
        N = X.shape[1]
        if self.batchnorm:
            _,_,_,P,_ = self.forward(X, testing) # (K,n)
        else:
            _,P = self.forward(X) # (K,n)
        #print("shape out",P.T.shape)
        k = np.argmax(P.T, axis=1)
        return np.sum(k == y) / N

    #def compute_grad_num(self, paramstr, X, Y, lam)

    # for batchnorm atm
    def compute_gradients_num(self, X, Y, lam):
        grads_W = []
        grads_b = []
        if self.batchnorm:
            grads_gam = []
            grads_beta = []

        h = np.float64(1e-7)
        for l in range(self.n_layers):
            # weights
            old_w = self.W(l)
            self.params['W'+str(l)] = old_w + h 
            cost1 = self.compute_cost(X, Y, lam)
            self.params['W'+str(l)] = old_w - h 
            cost2 = self.compute_cost(X, Y, lam)
            self.params['W'+str(l)] = old_w
            grad_W = (cost1-cost2) / (2*h)
            grads_W.append(grad_W)
            # bias
            #self.params['b'+str(l)] = self.initb(l)
            old_b = self.b(l)
            self.params['b'+str(l)] = old_b + h 
            cost1 = self.compute_cost(X, Y, lam)
            self.params['b'+str(l)] = old_b - h 
            cost2 = self.compute_cost(X, Y, lam)
            self.params['b'+str(l)] = old_b
            grad_b = (cost1-cost2) / (2*h)
            grads_b.append(grad_b)

        if self.batchnorm:
            for l in range(self.n_layers - 1):
                #self.params['gam'+str(l)] = self.initgamma(l)
                old_gam = self.gam(l)
                self.params['gam'+str(l)] = old_gam + h 
                cost1 = self.compute_cost(X, Y, lam)
                self.params['gam'+str(l)] = old_gam - h 
                cost2 = self.compute_cost(X, Y, lam)
                self.params['gam'+str(l)] = old_gam
                grad_gam = (cost1-cost2) / (2*h)
                grads_gam.append(grad_gam)
                #self.params['beta'+str(l)] = self.initbeta(l)
                old_beta = self.beta(l)
                self.params['beta'+str(l)] = old_beta + h 
                cost1 = self.compute_cost(X, Y, lam)
                self.params['beta'+str(l)] = old_beta - h 
                cost2 = self.compute_cost(X, Y, lam)
                self.params['beta'+str(l)] = old_beta
                grad_beta = (cost1-cost2) / (2*h)
                grads_beta.append(grad_beta)

        if self.batchnorm:
            return grads_W, grads_b, grads_gam, grads_beta
        return grads_W, grads_b

    # Y is one hot (K,N)
    def compute_gradients(self, X, Y, lam):
        N = X.shape[1]
        K = Y.shape[0]
        #m = self.b1.shape[0]
        grads_W = []
        grads_b = []

        # X_list contains input as well
        if self.batchnorm:
            S_list, Shat_list, X_list,P,(new_mus,new_vas) = \
                    self.forward(X,training=True) # [],(K, N),[],[]
            grads_gam = []
            grads_beta = []
        else:
            X_list,P = self.forward(X) # [] (K, N)

        # debugging stuff:
        #if np.isnan(P).any():
        #    self.print_params()
        #    sys.exit()
        
        # out layer
        G = -(Y - P) # (K, N)


        if self.batchnorm:
            idx = self.n_layers - 1
            X_l = X_list[idx]
            #print("X_l: ", X_l)
            # weights
            dLdW = 1. / N * G @ X_l.T
            grad_W = dLdW + 2.*lam*self.W(idx)
            grads_W.append(grad_W)
            # bias
            next_dim = self.dims[idx+1]
            dLdb = 1. / N * G @ np.ones(N)
            grad_b = np.reshape(dLdb, (next_dim,1))
            grads_b.append(grad_b)
            # propagate backwards
            G = self.W(idx).T @ G
            G = G * np.piecewise(X_l, [X_l <= 0, X_l > 0], [0,1])  #* Ind(H > 0) 

        if self.batchnorm:
            end = -1
            start = self.n_layers - 2
        else:
            end = 0
            start = self.n_layers - 1

        # GRADS
        for l in range(start, end, -1):
            if not self.batchnorm:
                # weights
                X_l = X_list[l]
                dLdW = 1. / N * G @ X_l.T
                grad_W = dLdW + 2.*lam*self.W(l)
                grads_W.append(grad_W)
                # bias
                next_dim = self.dims[l+1]
                dLdb = 1. / N * G @ np.ones(N)
                grad_b = np.reshape(dLdb, (next_dim,1))
                grads_b.append(grad_b)
            else:
                # grad gamma 
                #print("Layer ",l)
                Shat_l = Shat_list[l]
                S_l = S_list[l]
                grad_gam = 1./N * (G * Shat_l) @ np.ones(N)
                grads_gam.append(grad_gam)
                
                # grad beta
                grad_beta = 1./N * G @ np.ones(N)
                grads_beta.append(grad_beta)
                ones = np.ones((N,1))
                # prop backwards from scaling
                G = G * (self.gam(l) @ ones.T)

                # batch norm layers
                eps= np.finfo(np.float64).eps
                var = new_vas[l]
                sig1 = 1. / np.sqrt(var+eps)
                sig2 = 1. / np.power(var+eps, 1.5)

                G1 = G * (sig1 @ ones.T)
                G2 = G * (sig2 @ ones.T)
                #G1 = G * sig1 
                #G2 = G * sig2 

                mu = new_mus[l]
                D = S_l - mu @ ones.T
                #D = S_l - mu
                c = (G2 * D) @ ones

                G = G1 - 1./N*(G1 @ ones) @ ones.T - 1./N * D *(c @ ones.T)

                # grads for W and bias
                X_l = X_list[l]

                dLdW = 1. / N * G @ X_l.T
                grad_W = dLdW + 2.*lam*self.W(l)
                grads_W.append(grad_W)
                next_dim = self.dims[l+1]
                dLdb = 1. / N * G @ np.ones(N)
                grad_b = np.reshape(dLdb, (next_dim,1))
                grads_b.append(grad_b)

            # propagate backwards
            if l > 0:
                G = self.W(l).T @ G
                G = G * np.piecewise(X_l, [X_l <= 0, X_l > 0], [0,1])  #* Ind(H > 0) 
        # in layer weights
        if not self.batchnorm:
            dLdW0 = 1. / N * G @ X.T
            grad_W0 = dLdW0 + 2.*lam*self.W(0)
            grads_W.append(grad_W0)
            # in layer bias
            next_dim = self.dims[1]
            dLdb0 = 1. / N * G @ np.ones(N)
            grad_b0 = np.reshape(dLdb0, (next_dim,1))
            grads_b.append(grad_b0)

        if self.batchnorm:
            return grads_W, grads_b, grads_gam, grads_beta
        return grads_W, grads_b
    
    

    # X: cols are data points
    def training(self, X,Y,X_val, Y_val, lam=0, n_batch=100, n_epochs=20,
            eta_min=1e-5, eta_max=1e-1,n_s=500, print_epoch=5):
        costs_train = []
        costs_val = []
        N = X.shape[1]

        eta = eta_min
        t = 1
        
        for epoch in range(n_epochs):
            # shuffle
            permidx = np.random.permutation(N)
            Xtrain = X[:,permidx]   
            Ytrain = Y[:,permidx]   

            # TRAIN
            for j in range(int(N / n_batch)):
                # get batch
                j_start = j*n_batch #inclusive start
                j_end = (j+1)*n_batch # exclusive end
                Xbatch = Xtrain[:, j_start:j_end]
                Ybatch = Ytrain[:, j_start:j_end]

                # get gradients, opposite order from last to first
                if self.batchnorm:
                    grads_W, grads_b, grads_gam, grads_beta = \
                            self.compute_gradients(Xbatch,Ybatch,lam)
                else:
                    grads_W, grads_b = \
                            self.compute_gradients(Xbatch,Ybatch,lam)

                # UPDATES
                for l in range(self.n_layers):
                    gradw = grads_W[self.n_layers-1-l]
                    #print("gradw {}, iter {}".format(gradw,j))
                    self.params["W"+str(l)] -= \
                            eta * gradw 
                    gradb = grads_b[self.n_layers-1-l]
                    #print("gradb {}, iter {}".format(gradb,j))
                    self.params["b"+str(l)] -= \
                            eta * gradb 
                if self.batchnorm:
                    # only for hidden layer(s)
                    for hl in range(self.n_layers - 1):
                        gamgrad = grads_gam[self.n_layers-2-hl]
                        gamgrad = gamgrad.reshape((len(gamgrad),1))
                        #print("gradgam {}, iter {}".format(gamgrad,j))
                        self.params["gam"+str(hl)] -= \
                                eta * gamgrad 
                        betagrad = grads_beta[self.n_layers-2-hl]
                        betagrad = betagrad.reshape((len(betagrad),1))
                        #print("gradbeta {}, iter {}".format(betagrad,j))
                        self.params["beta"+str(hl)] -= \
                                eta * betagrad

                #self.W1 = self.W1 - eta * grad_W1
                #self.b1 = self.b1 - eta * grad_b1

                # adjust cyclic learn rate
                if t <= n_s:
                    eta = eta_min + t / n_s * (eta_max - eta_min)
                if n_s < t and t <= 2*n_s:
                    eta = eta_max - (t-n_s)/n_s * (eta_max-eta_min)
                    if t == 2*n_s:
                        t = 1
                t+=1

            # save costs
            # TODO: USE MOVING MU AND VAR HERE
            costtrain = self.compute_cost(X, Y, lam)
            costs_train.append(costtrain)
            costval = self.compute_cost(X_val, Y_val, lam)
            costs_val.append(costval)
            
            # print epoch info
            if epoch % print_epoch == 0:
                print("Epoch {} ; traincost: {} ; valcost: {}".format(epoch, costtrain, costval))
        return costs_train, costs_val


# ------- END CLASS

# compare numerical and analytical gradient with one cifar batch
def check_grads(cifar_10_dir):
    print("Checking gradients....")
    train_data, train_filenames, train_labels, train_onehot,\
    val_data, val_filenames, val_labels, val_onehot,\
    test_data, test_filenames, test_labels, label_names, test_onehot = \
        load_cifar_10_data_onebatch(cifar_10_dir)

    # preprocess
    mean_train = np.mean(train_data)
    std_train = np.std(train_data)
    train_data = train_data - mean_train
    train_data = train_data / std_train
    train_data = train_data.T
    print("train dat shape: ", train_data.shape)
    print("train labesl shape: ", train_labels.shape)


    Xbatch = train_data[:, 5:15]
    Ybatch = train_labels[5:15]
    K = 10 # classes
    d = 3072 # input dim
    dims = [d,50,50,K] 
    #dims = [d,50,K] 
    #net = Net(dims, batchnorm = True)
    net = Net(dims, batchnorm = False)
    lam = 0

    # analytical
    grads_an = net.compute_gradients(Xbatch,Ybatch,lam)
    # numerical
    grads_num = net.compute_gradients_num(Xbatch,Ybatch,lam)
    print("comparing...")
    compare_grads(grads_an, grads_num, net.n_layers)

# get max relative err between analytical and num grads
# assumes batch norm
def compare_grads(grads_an, grads_num, n_layers):
    #grads_W_an, grads_b_an, grads_gam_an, grads_beta_an = grads_an
    #grads_W_num, grads_b_num, grads_gam_num, grads_beta_num = grads_num
    grads_W_an, grads_b_an = grads_an
    grads_W_num, grads_b_num = grads_num

    for l in range(n_layers):
        # weights
        wstr = 'W'+str(l)
        w_err = graddiff(wstr, grads_W_an[l], grads_W_num[l])
        print("param {}, max rel err: {}".format(wstr, w_err))
        # bias
        bstr = 'b'+str(l)
        b_err = graddiff(bstr, grads_b_an[l], grads_b_num[l])
        print("param {}, max rel err: {}".format(bstr, b_err))
            
    '''
    for l in range(n_layers - 1):
        # gamma
        gamstr = 'gam'+str(l)
        gam_err = graddiff(gamstr, grads_gam_an[l], grads_gam_num[l])
        print("param {}, max rel err: {}".format(gamstr, gam_err))
        # beta
        betastr = 'beta'+str(l)
        beta_err = graddiff(betastr, grads_beta_an[l], grads_beta_num[l])
        print("param {}, max rel err: {}".format(betastr, beta_err))
    '''
    print("finito...")


def graddiff(paramstr, grad_an, grad_num):
    n = abs(grad_an.flat[:] - grad_num.flat[:])
    d = np.asarray([max(abs(fa), 1e-10+abs(fn)) for fa,fn in \
            zip(grad_an.flat[:], grad_num.flat[:])])
    # get the maxium error relative
    return max(n / d)
            

def main():
    np.random.seed(400)
    cifar_10_dir = 'Dataset/cifar-10-batches-py'

    N_val = 5000

    train_data, train_filenames, train_labels, train_onehot,\
    val_data, val_filenames, val_labels, val_onehot,\
    test_data, test_filenames, test_labels, label_names, test_onehot = \
        load_cifar_10_data(cifar_10_dir, N_val)

    print("Train data: ", train_data.shape)
    print("Train filenames: ", train_filenames.shape)
    print("Train labels: ", train_labels.shape)
    print("Train onehot: ", train_onehot.shape)
    print("Val data: ", val_data.shape)
    print("val filenames: ", val_filenames.shape)
    print("val labels: ", val_labels.shape)
    print("val onehot: ", val_onehot.shape)
    print("Test data: ", test_data.shape)
    print("Test filenames: ", test_filenames.shape)
    print("Test labels: ", test_labels.shape)
    print("test onehot: ", test_onehot.shape)
    print("Label names: ", label_names.shape)

    # Pre-process
    mean_train = np.mean(train_data)
    std_train = np.std(train_data)
    #print("mean, std of train: ", mean_train, " ; ", std_train)
    train_data = train_data - mean_train
    train_data = train_data / std_train
    val_data = val_data - mean_train
    val_data = val_data / std_train
    test_data = test_data - mean_train
    test_data = test_data / std_train

    train_data = train_data.T
    val_data = val_data.T
    test_data = test_data.T

    
    ###### TRAINING #####

    # Epochs and batches
    N = train_data.shape[0]
    n_batch=100
    #n_batch=50
    #n_epochs=21

    # LAMBDAS search 
    #l_middle = np.log10(0.004730593550311557)
    #l_min = -5
    #l_min = l_middle - 0.5
    #l_max = -1
    #l_max = l_middle + 0.5
    #n_lambdas = 6
    #lambdas = np.power(10,np.random.uniform(low=l_min,high=l_max,size=(n_lambdas,)))
    lambdas = [0.006976611574964777]

    # Cyclic Learning rate
    #epochs_per_cycle = 2
    #n_s = epochs_per_cycle*np.floor(N / n_batch)
    #n_s =  5*45000 / n_batch
    n_s =  2*45000 / n_batch
    eta_min = 1e-5
    eta_max = 1e-1
    n_epochs = 8

    # Network params
    K = 10 # classes
    d = 3072 # input dim
    dims = [d,50,50,K] 
    #dims = [d,50,30,20,20,10,10,10,10,K] 
    #m = 130 # hid

    # changes during lambda search
    best_valcost = 10000
    best_valacc = -1
    best_testacc = -1
    best_lam = lambdas[0]
    best_net = 0
    #nets = []
    best_trainloss = []
    best_valloss = []

    # LOOP TRAIN
    for lidx, lam in enumerate(lambdas):
        print("----")
        print("Trying lambda: ", lam)
        net = Net(dims, batchnorm = True)
        
        #net = Net(dims)

        #net.print_params_shape()
        #X_list, P = net.forward(train_data)
        #X_list, P, (mus, vas) = net.forward(train_data,training=True)
        #c = net.compute_cost(train_data, train_onehot, lam) 
        #a = net.compute_accuracy(train_data, train_onehot) 

        costs_train, costs_val = net.training(
            train_data,
            train_onehot,
            val_data, 
            val_onehot,
            #eta=eta, 
            lam=lam, 
            n_batch=n_batch, 
            n_epochs=n_epochs,
            eta_min=eta_min,
            eta_max=eta_max,
            n_s = n_s,
            print_epoch = 3,
        )
        

        #nets.append(copy(net))
        last_valcost = costs_val[-1]
        val_acc = net.compute_accuracy(val_data, val_labels)
        test_acc = net.compute_accuracy(test_data, test_labels)

                 
        # print and log stuff
        print("DONE lam {}, valcost {}, valacc {}, testacc {}"\
                .format(lam, last_valcost, val_acc, test_acc))
        with open("lambdas.txt","a") as f:
            f.write("lam {}, val_cost {}, val_acc {}, test_acc {}\n"\
                    .format(lam, last_valcost, val_acc, test_acc))

        # check if better than previous lambdas
        #if  last_valcost < val_cost:
        if val_acc > best_valacc:
            best_valacc = val_acc
            best_net = lidx 
            best_valcost = last_valcost
            best_lam = lam
            best_testacc = test_acc
            best_trainloss = costs_train
            best_valloss = costs_val

    # print and log test acc for the best
    #test_acc = nets[best_net].compute_accuracy(test_data, test_labels)
    print(" ------")
    print("BEST VALIDATION lam #{}: {}, test acc {}"\
            .format(best_net,best_lam, best_testacc))
    with open("lambdas.txt","a") as f:
        f.write("best validation lam #{}: {}, test acc {}\n---------\n"\
                .format(best_net, lam, best_testacc))

    # plot the validation and train errs
    fig = plt.figure()
    fig.suptitle(
        'train (red) and val (green) cross entropy cost, \nlambda={}'.format(lam),
        fontsize=16
    )
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('cost', fontsize=14)
    plt.plot(best_trainloss, 'r')
    plt.plot(best_valloss, 'g')
    plt.show()


if __name__ == "__main__":
    main()
