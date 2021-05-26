import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.stats import mode
from copy import deepcopy,copy
import sys
from collections import OrderedDict
import math

def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)
def relu(x):
    """ Standard definition of the relu function """
    return np.maximum(0,x)
def tanh(x):
    return np.tanh(x)

def get_data(filename):
    """ 
    Fetches data from text file 'filename' and returns:

    book_data: all the text in the text file as a string
    char2idx: character to index mapping
    idx2char: idx to character mapping
    num_chars: the size of the character vocabulary
    """
    print("Reading data...")
    with open(filename, 'r', encoding='utf-8') as f:
        book_data = f.read().replace('\n', '')
    char2idx = {}
    idx2char = {}
    idx = 0
    print("Constructing c2i and i2c...")
    for c in book_data:
        if c not in char2idx.keys():
            char2idx[c] = idx
            idx2char[idx] = c
            idx += 1
        
    num_chars = idx+1 # later set to K
    return book_data, char2idx, idx2char, num_chars
     
class RNN:
    # m: hidden dim
    # K: vocab size 
    def __init__(self, m, K, char2idx, idx2char, seq_max_length=25):
        self.m = m
        self.d = K # in
        self.K = K # out
        self.seq_max_length = 25

        # weights and biases
        sig = 0.01
        mu = 0
        self.U = np.random.normal(mu,sig, (m, K)) # (m,K)
        self.grad_U = np.zeros((m,K))
        self.W = np.random.normal(mu,sig, (m, m)) # (m,m)
        self.grad_W = np.zeros((m,m))
        self.V = np.random.normal(mu,sig, (K, m)) # (K,m)
        self.grad_V = np.zeros((K,m))

        self.b = np.zeros((m,1)) # (m,1)
        self.grad_b = np.zeros((m,1))
        self.c = np.zeros((K,1)) # (K,1)
        self.grad_c = np.zeros((K,1))

        # char and idx mappings
        self.char2idx = char2idx
        self.idx2char = char2idx
        self.vocab_size = K 

    def char_to_onehot(self, char):
        """
        input: a character
        returns: one hot encoding numpy vector (1, K)
        """
        one_hot = np.zeros((self.vocab_size,1), dtype=int)  
        idx = self.char2idx[char]
        one_hot[idx] = 1
        return one_hot.T # (1, K)

    def onehot_to_char(self, onehot_vect):
        """
        input: one hot vector (1,K)
        output: related character to the onehot vector 
        """
        idx = np.argwhere(onehot_vect == 1)[0][1]
        return self.idx2char[idx]

    def forward(self, X):
        """
        X: [x1,x2,...xt], each x column one-hot vectors
        """
        h_t = np.zeros((self.m,1)) # set to initial h_0
        T = X.shape[1]
        A = np.zeros((self.m, T)) # activation inputs
        H = np.zeros((self.m, T)) # hidden states
        P = np.zeros((self.K, T)) # softmax outputs
        for t in range(T):
            # compute values
            x_t = X[:,t].reshape(self.K,1)
            a_t = self.W @ h_t + self.U @ x_t + self.b # (m,1)
            h_t = tanh(a_t) # (m,1)
            o_t = self.V @ h_t + self.c # (K,1)
            p_t = softmax(o_t) # (K,1)
            # save values
            A[:,t] = a_t.reshape(self.m)
            H[:,t] = h_t.reshape(self.m)
            P[:,t] = p_t.reshape(self.K)
        return P, H, A

    def compute_cost(self,P,Y):
        # P (K, num_steps) probs for each character at each time step
        # Y (K, num_steps) one hots for every row
        T = P.shape[1] 
        loss = 0
        for i in range(T):
            p_t = P[:,t].reshape(self.K,1) # probs for each char
            y_t = Y[:,t].reshape(self.K,1) # char onehot label at time t
            l_t += -np.log(np.dot(y_t.T,p_t)) # cross entropy loss
            loss += l_t
        return loss
    
    def generate(self, x0, h0, N):
        # x0: init char one-hot (K,1)
        # h0: init hidden vects (m,1)
        # N: length of generated sequence, i.e num of characters
        Y = np.zeros((self.K, N)) # one-hots for each generated character
        x_n = x0 
        h_n = h0
        for n in range(N):
            # x (K, 1) 
            # p (K, 1)
            # h (m, 1)
            a_t = self.W @ h_t + self.U @ x_t + self.b # (m,1)
            h_t = tanh(a_t) # (m,1)
            o_t = self.V @ h_t + self.c # (K,1)
            p_t = softmax(o_t) # (K,1)
            choice_idx = np.random.choice(
                a=self.K, 
                size=1,
                p=p_t.reshape(self.K)
            )
            # make one-hot vector
            x_n = np.zeros((self.K,1))
            x_n[choice_idx] = 1
            # save generated output
            Y[:,n] = x_n.reshape(self.K)
        return Y

    def compute_gradients(self, X, Y, lam):
        N = X.shape[1]
        K = Y.shape[0]
        grads_W = []
        grads_b = []

    def training(self, book_data, lr=0.1, epochs=2, print_loss=100,
            print_generate=10000)
        N = len(book_data)
        n_sequences = math.floor(N / self.seq_max_length)

        it = 0

        # EPOCHS
        for epoch in tqdm(range(epochs)):
            e = 0 # 0-indexed in python, no matlab here :)
            h0 = np.zeros((self.m,1))
            # ITERATING EACH SEQUENCE i
            for i in range(n_sequences):
                # (K,seq_max_len)
                X_chars = book_data[e:e+self.seq_max_length]
                X = 
                # (K,seq_max_len)
                Y_chars = book_data[e+1:e+self.seq_max_length+1] 
                
                # TODO
                grads_W, grads_b = \
                    self.compute_gradients(Xbatch,Ybatch,lam)

                # UPDATES TODO
                gradw = grads_W[self.n_layers-1-l]
                self.params["W"+str(l)] -= \
                    eta * gradw 
            
            # save loss TODO
            costtrain = self.compute_cost(X, Y, lam)
            costs_train.append(costtrain)
        
            # save best model TODO
            
            # print loss info TODO
            if epoch % print_epoch == 0:
                #print("Epoch {} ; traincost: {} ; valcost: {}".format(epoch, costtrain, costval))

            # print generation TODO

        # TODO what to return?
        return costs_train, costs_val

# ------- END CLASS

def main():
    np.random.seed(400)
    filename = "goblet_book.txt"
    book_data, char2idx, idx2char, K = get_data(filename)
    print("char2idx: ", char2idx)

    print("Initialising model...")
    m = 100
    seq_max_length = 25
    net = RNN(m, K, char2idx, idx2char,seq_max_length)

    lr = 0.1
    epochs = 2

    """
    print("Training beginning...")
    net.train(
        data = book_data
        lr = lr
        epochs = epochs
    )
    """
    
if __name__ == "__main__":
    main()
