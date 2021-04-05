import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.io as sio

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


def load_cifar_10_data(data_dir, negatives=False):
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


class Net2:
    # d: input dim,
    # m: hidden dim
    # K: out dim
    def __init__(self, d,m, K):
        std1 = 1. / np.sqrt(d)
        self.W1 = np.random.normal(0,std1,(m,d))
        #self.b1 = np.zeros((m,1)) 
        self.b1 = np.random.normal(0,.1,(m,1)) 
        std2 = 1. / np.sqrt(m)
        self.W2 = np.random.normal(0,0.1,(K, m))
        self.b2 = np.random.normal(0,.1,(K,1)) 
        #self.b2 = np.zeros((K,1)) 
        #xavi_std = np.sqrt(2. / (d+K))
        #self.W = np.random.normal(0,xavi_std,(K,d)) # xavier

    # returns output and hidden output
    def forward(self,X):
        S1 = self.W1 @ X + self.b1
        H = relu(S1)
        S = self.W2 @ H + self.b2
        return H, softmax(S)

    # Y: one-hot, (K, n)
    # X: data are cols
    # lam: penalty term
    # returns cross-entropy loss
    def compute_cost(self,X, Y, lam):
        _,P = self.forward(X) # (K,n)
        N = X.shape[1]
        l = -Y*np.log(P) # categorical cross-entropy
        return 1. / N * np.sum(l) + lam * \
                (np.sum(self.W1**2) + np.sum(self.W2**2))

    def compute_accuracy(self, X, y):
        N = X.shape[1]
        _,P = self.forward(X) # (K,N)
        print("shape out",P.T.shape)
        k = np.argmax(P.T, axis=1)
        return np.sum(k == y) / N

    # Y is one hot (K,N)
    def compute_gradients(self, X, Y, lam):
        N = X.shape[1]
        K = Y.shape[0]
        m = self.b1.shape[0]
        H,P = self.forward(X) # (K, N)
        # out
        G = -(Y - P) # (K, N)
        # hid
        dLdW2 = 1. / N * G @ H.T
        dLdb2 = 1. / N * np.matmul(G,np.ones(N))
        dLdb2 = 1. / N * G @ np.ones(N)

        # activ in
        G = self.W2.T @ G
        G = G * np.piecewise(H, [H <= 0, H > 0], [0,1])  #* Ind(H > 0) 
        # input layer
        dLdW1 = 1. / N * G @ X.T
        dLdb1 = 1. / N * G @ np.ones(N)
        
        # gradients
        grad_W2 = dLdW2 + 2.*lam*self.W2
        grad_b2 = np.reshape(dLdb2, (K,1))
        grad_W1 = dLdW1 + 2.*lam*self.W1
        grad_b1 = np.reshape(dLdb1, (m,1))
        '''
        dLdW = 1. / N * G @ X.T
        dLdb = 1. / N * np.matmul(G,np.ones(N))
        grad_W = dLdW + 2.*lam*self.W
        grad_b = np.reshape(dLdb, (K,1))
        '''
        return grad_W1, grad_b1, grad_W2, grad_b2
    
    # X: cols are data points
    def training(self, X,Y,X_val, Y_val, eta=0.001, lam=0, n_batch=100, n_epochs=20):
        print("training started...")
        costs_train = []
        costs_val = []
        N = X.shape[1]
        
        for epoch in range(n_epochs):
            eta = eta * 1/(1+ 0.09*epoch) 
            if eta < 0.001:
                eta = 0.001
            # shuffle
            permidx = np.random.permutation(N)
            Xtrain = X[:,permidx]   
            Ytrain = Y[:,permidx]   
            for j in range(int(N / n_batch)):
                j_start = j*n_batch #inclusive start
                j_end = (j+1)*n_batch # exclusive end
                Xbatch = Xtrain[:, j_start:j_end]
                Ybatch = Ytrain[:, j_start:j_end]
                grad_W1, grad_b1, grad_W2, grad_b2 = \
                        self.compute_gradients(Xbatch, Ybatch, lam)
                # update params
                self.W2 = self.W2 - eta * grad_W2
                self.b2 = self.b2 - eta * grad_b2
                self.W1 = self.W1 - eta * grad_W1
                self.b1 = self.b1 - eta * grad_b1
            # save costs
            costtrain = self.compute_cost(X, Y, lam)
            costs_train.append(costtrain)
            costval = self.compute_cost(X_val, Y_val, lam)
            costs_val.append(costval)
            #if epoch % 10 == 0:
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

    train_data, train_filenames, train_labels, train_onehot,\
    val_data, val_filenames, val_labels, val_onehot,\
    test_data, test_filenames, test_labels, label_names, test_onehot = \
        load_cifar_10_data(cifar_10_dir)

    print("Train data: ", train_data.shape)
    print("Train filenames: ", train_filenames.shape)
    print("Train labels: ", train_labels.shape)
    print("Train onehot: ", train_onehot)
    print("Val data: ", val_data.shape)
    print("val filenames: ", val_filenames.shape)
    print("val labels: ", val_labels.shape)
    print("val onehot: ", val_onehot.shape)
    print("Test data: ", test_data.shape)
    print("Test filenames: ", test_filenames.shape)
    print("Test labels: ", test_labels.shape)
    print("test onehot: ", test_onehot.shape)
    print("Label names: ", label_names.shape)

    K = 10
    m = 50
    d = 3072
    N = 10000 # for train,val,test

    # pre-process
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

    # 3.init model
    np.random.seed(400)
    net = Net2(d,m,K)

    # 4.evaluate
    #P1 = net.forward(train_data[:,0:15]) # K x n
    # 5. compute cost
    #lam = 0.01
    #cost1 = net.compute_cost(train_data, train_onehot, lam)
    #print("COST1",cost1)
    #acc1 = net.compute_accuracy(train_data, train_labels)
    #print("ACC1",cost1)
    #grad_W1, grad_b1 = net.compute_gradients(train_data, train_onehot, lam)
    #errW1,errb1 = net.compare_grad(train_data,train_onehot, lam)

    # TRAINING
    eta = 0.01
    lam=0
    n_batch=100
    n_epochs=15
    costs_train, costs_val = net.training(
        train_data,
        train_onehot,
        val_data, 
        val_onehot,
        eta=eta, 
        lam=lam, 
        n_batch=n_batch, 
        n_epochs=n_epochs
    )
    test_acc = net.compute_accuracy(test_data, test_labels)
    print("\n")
    print("PARAMS: eta={}, lam={}, n_batch={}, n_epochs={}".format(eta,lam,n_batch,n_epochs))
    print("TEST ACCURACY: ", test_acc)
    '''

    # plot the validation and train errs
    fig = plt.figure()
    fig.suptitle(
        'train (red) and val (green) cross entropy cost, \ntest acc={}'.format(test_acc),
        fontsize=18
    )
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('cost', fontsize=14)
    plt.plot(costs_train, 'r')
    plt.plot(costs_val, 'g')

    # plot visualization of weights
    fig2 = plt.figure()
    fig2.suptitle(
        "weight visualizations",
        fontsize=18
    )
    cols = 5
    rows = 2
    for row in range(rows):
        for col in range(cols):
            i = row*cols + col
            im = np.reshape(net.W[i,:], (32,32,3))
            s_im = (im - np.min(im)) / (np.max(im) - np.min(im))
            fig2.add_subplot(rows,cols,i+1)
            plt.imshow(s_im)

    plt.show()
    '''

if __name__ == "__main__":
    main()
