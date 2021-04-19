import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.stats import mode
from copy import deepcopy,copy

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
        self.n_layers = len(dims)-1 # -2 for #hidden
        self.batchnorm = batchnorm
        if batchnorm:
            self.mus = []
            self.vas = []
        # W: outdim x indim
        # b: outdim x 1
        for l in range(self.n_layers):
            self.params['W'+str(l)] = self.he_initW(l)
            self.params['b'+str(l)] = self.initb(l)
            if batchnorm:
                self.params['gam'+str(l)] = self.initgamma(l)
                self.params['beta'+str(l)] = self.initbeta(l)

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

    def rand_initW(self, layer):
        return np.random.normal(
            0,0.1, 
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
    def forward(self,X_in, training=False, testing=False,mus=[], vas=[]):
        X_list = [] 
        X_l = X_in
        N, _ = X_in.shape
        new_mus = []
        new_vas = []
        
        # all layers except out layer
        for l in range(self.n_layers-1):
            S_l = self.W(l) @ X_l + self.b(l)
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
                    var = (N-1)/N * np.var(S_l, axis=1, keepdims=True) 
                    new_vas.append(var)
                    if training:
                        a = 0.8 # alpha
                        self.mus[l] = a * self.mus[l] + (1-a) * mu 
                        self.vas[l] = a * self.vas[l] + (1-a) * var
                # batch normalize
                eps= np.finfo(np.float64).eps
                vareps = var + eps
                S_l = 1. / np.sqrt(np.diag(vareps)) * (S_l - mu)
                # scale
                S_l = self.gam(l) * S_l + self.beta(l)
            # activation function
            X_l = relu(S_l)
            X_list.append(X_l)
            
        # output layer
        S = self.W(self.n_layers-1) @ X_l + self.b(self.n_layers-1)
        P = softmax(S)
        if self.batchnorm:
            return X_list, P, (new_mus, new_vas)
        return X_list, P

    # Y: one-hot, (K, n)
    # X: data are cols
    # lam: penalty term
    # returns cross-entropy loss
    def compute_cost(self,X, Y, lam):
        if self.batchnorm:
            _,P,_ = self.forward(X) # (K,n)
        else:
            _,P = self.forward(X) # (K,n)
        N = X.shape[1]
        crossl = -Y*np.log(P) # categorical cross-entropy
        reg = 0
        for l in range(self.n_layers - 1):
            reg += np.sum(self.W(l)**2)
        loss = 1. / N * np.sum(crossl) + lam * reg
        return loss

    def compute_accuracy(self, X, y):
        N = X.shape[1]
        if self.batchnorm:
            _,P,_ = self.forward(X) # (K,n)
        else:
            _,P = self.forward(X) # (K,n)
        #print("shape out",P.T.shape)
        k = np.argmax(P.T, axis=1)
        return np.sum(k == y) / N

    # Y is one hot (K,N)
    def compute_gradients(self, X, Y, lam):
        N = X.shape[1]
        K = Y.shape[0]
        #m = self.b1.shape[0]
        if self.batchnorm:
            X_list,P,new_mus,new_vas = self.forward(X) # [],(K, N),[],[]
        else:
            X_list,P = self.forward(X) # [] (K, N)
        grads_W = []
        grads_b = []

        # out layer
        G = -(Y - P) # (K, N)

        # loop through hidden layers
        for l in range(self.n_layers - 1, 0, -1):
            # weights
            X_l = X_list[l-1]
            dLdW = 1. / N * G @ X_l.T
            grad_W = dLdW + 2.*lam*self.W(l)
            grads_W.append(grad_W)
            # bias
            next_dim = self.dims[l+1]
            dLdb = 1. / N * G @ np.ones(N)
            grad_b = np.reshape(dLdb, (next_dim,1))
            grads_b.append(grad_b)

            # propagate backwards
            G = self.W(l).T @ G
            G = G * np.piecewise(X_l, [X_l <= 0, X_l > 0], [0,1])  #* Ind(H > 0) 
        # in layer weights
        dLdW0 = 1. / N * G @ X.T
        grad_W0 = dLdW0 + 2.*lam*self.W(0)
        grads_W.append(grad_W0)
        # in layer bias
        next_dim = self.dims[1]
        dLdb0 = 1. / N * G @ np.ones(N)
        grad_b0 = np.reshape(dLdb0, (next_dim,1))
        grads_b.append(grad_b0)

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

            for j in range(int(N / n_batch)):
                # get batch
                j_start = j*n_batch #inclusive start
                j_end = (j+1)*n_batch # exclusive end
                Xbatch = Xtrain[:, j_start:j_end]
                Ybatch = Ytrain[:, j_start:j_end]
                # get gradients, opposite order from last to first
                grads_W, grads_b = self.compute_gradients(Xbatch,Ybatch,lam)

                # update params
                for l in range(self.n_layers):
                    self.params["W"+str(l)] -= \
                            eta * grads_W[self.n_layers-1-l]
                    self.params["b"+str(l)] -= \
                            eta * grads_b[self.n_layers-1-l]

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
            costtrain = self.compute_cost(X, Y, lam)
            costs_train.append(costtrain)
            costval = self.compute_cost(X_val, Y_val, lam)
            costs_val.append(costval)
            
            # print epoch info
            if epoch % print_epoch == 0:
                print("Epoch {} ; traincost: {} ; valcost: {}".format(epoch, costtrain, costval))
        return costs_train, costs_val

    def compare_grad(self,X,Y, lam):
        P = self.forward(X)
        grad_W, grad_b = self.compute_gradients(X, Y, lam)
        h = 1e-6
        grad_Wnum, grad_bnum = ComputeGradsNumSlow(X, Y, P, self.W, self.b, lam, h)
        '''
        print(grad_W.shape)
        print(grad_b.shape)
        print(grad_Wnum.shape)
        print(grad_bnum.shape)
        '''
        errW = np.abs(grad_W - grad_Wnum) / np.max(1e-6, np.abs(grad_W)+np.abs(grad_Wnum))
        errb = np.abs(grad_b - grad_bnum) / np.max(1e-6, np.abs(grad_b)+np.abs(grad_bnum))
        return errW, errb


def main():
    cifar_10_dir = 'Dataset/cifar-10-batches-py'

    N_val = 1000

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
    print("mean, std of train: ", mean_train, " ; ", std_train)
    train_data = train_data - mean_train
    train_data = train_data / std_train
    val_data = val_data - mean_train
    val_data = val_data / std_train
    test_data = test_data - mean_train
    test_data = test_data / std_train

    train_data = train_data.T
    val_data = val_data.T
    test_data = test_data.T

    np.random.seed(400)
    
    ###### TRAINING #####

    # Epochs and batches
    N = train_data.shape[0]
    n_batch=100
    #n_batch=50
    n_epochs=9

    # Lambda search 
    l_min = -7
    l_max = -5
    n_lambdas = 6
    #lambdas = np.power(10,np.random.uniform(low=l_min,high=l_max,size=(n_lambdas,)))
    lambdas = [0.005]

    # Cyclic Learning rate
    epochs_per_cycle = 2
    #n_s = epochs_per_cycle*np.floor(N / n_batch)
    n_s =  5*45000 / n_batch
    eta_min = 1e-5
    eta_max = 1e-1

    # Network params
    K = 10 # classes
    d = 3072 # input dim
    dims = [d,50,50,K] 
    #dims = [d,50,30,20,20,10,10,10,10,K] 
    #m = 130 # hid

    # changes during lambda search
    best_val = -1 
    best_lam = lambdas[0]
    best_net = 0
    nets = []

    # LOOP TRAIN
    for lidx, lam in enumerate(lambdas):
        print("----")
        print("Trying lambda: ", lam)
        net = Net(dims, batchnorm = True)

        #net.print_params_shape()
        #X_list, P = net.forward(train_data)
        #X_list, P, (mus, vas) = net.forward(train_data,training=True)
        #c = net.compute_cost(train_data, train_onehot, lam) 
        #a = net.compute_accuracy(train_data, train_onehot) 

        return
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
            print_epoch = 2,
        )
        nets.append(copy(net))
        val_acc = net.compute_accuracy(val_data, val_labels)

                 
        # print and log stuff
        last_valcost = costs_val[-1]
        print("DONE lam {}, valcost {}, valacc {}"\
                .format(lam, last_valcost, val_acc))
        with open("lambdas.txt","a") as f:
            f.write("lam {}, val_cost {}, val_acc\n"\
                    .format(lam, last_valcost, val_acc))

        # check if better than previous lambdas
        #if  last_valcost < val_cost:
        if val_acc > best_val:
            best_valacc = val_acc
            best_net = lidx 
            best_valcost = last_valcost
            best_lam = lam

    # print and log test acc for the best
    test_acc = nets[best_net].compute_accuracy(test_data, test_labels)
    print(" ------")
    print("BEST lam #{}: {}, test acc {}"\
            .format(best_net,best_lam, test_acc))
    with open("lambdas.txt","a") as f:
        f.write("best lam #{}: {}, test acc {}\n"\
                .format(best_net, lam, test_acc))

    # plot the validation and train errs
    fig = plt.figure()
    fig.suptitle(
        'train (red) and val (green) cross entropy cost, \nlambda={}'.format(lam),
        fontsize=16
    )
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('cost', fontsize=14)
    plt.plot(costs_train, 'r')
    plt.plot(costs_val, 'g')
    plt.show()


if __name__ == "__main__":
    main()
