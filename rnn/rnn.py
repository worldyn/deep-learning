import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.stats import mode
from copy import deepcopy,copy
import sys
from collections import OrderedDict
import math
from tqdm import tqdm

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
    #forbiddens = ['\t', 'ü','{','}','_','/','^','•','\\']
    forbiddens = []
    for c in book_data:
        if c not in char2idx.keys() and c not in forbiddens:
            char2idx[c] = idx
            idx2char[idx] = c
            idx += 1
        
    return book_data, char2idx, idx2char
     
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
        self.idx2char = idx2char 
        self.vocab_size = K 

        # adagrads params (save in case you run training several times)
        self.m_U = np.zeros((m, K))
        self.m_W = np.zeros((m, m))
        self.m_V = np.zeros((K, m))
        self.m_b = np.zeros((m, 1))
        self.m_c = np.zeros((K, 1))

    
    def weigths_dict(self):
        return {"U": self.U, "W": self.W, 
                "V": self.V, "b": self.b, "c": self.c}

    def load_weigths(self, wd):
        # wd: weigths dict as in weights_dict(self) above
        self.U = wd["U"]
        self.W = wd["W"]
        self.V = wd["V"]
        self.b = wd["b"]
        self.c = wd["c"]

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
        idx = int(np.argwhere(onehot_vect == 1)[0][1])
        return self.idx2char[idx]

    def forward(self, X, hprev):
        """
        X: [x1,x2,...xt], each x column one-hot vectors
        """
        h_t = hprev 
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

    def compute_loss(self,P,Y):
        # P (K, num_steps) probs for each character at each time step
        # Y (K, num_steps) one hots for every row
        T = P.shape[1] 
        loss = 0
        for i in range(T):
            p_t = P[:,i].reshape(self.K,1) # probs for each char
            y_t = Y[:,i].reshape(self.K,1) # char onehot label at time t
            l_t = -np.log(np.dot(y_t.T,p_t)) # cross entropy loss
            loss += l_t
        return loss[0][0] # TODO fix dims
    
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
            a_t = self.W @ h_n + self.U @ x_n + self.b # (m,1)
            h_t = tanh(a_t) # (m,1)
            o_t = self.V @ h_n + self.c # (K,1)
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

    def compute_gradients(self, P, X, Y, H, H0, A):
        # grad w.r.t loss
        dLdout = -(Y.T - P.T).T
        # V and c
        self.grad_V = dLdout @ H.T
        #axis = 2
        self.grad_c = np.sum(dLdout, axis=1, keepdims=True)
        # w.r.t h and a (for each time step)
        T = X.shape[1]
        dLdH = np.zeros((T, self.m))
        dLdH[-1] = dLdout.T[-1] @ self.V # w.r.t last hidden state
        dLdA = np.zeros((self.m, T)) 
        # w.r.t last activation grad
        dLdA[:,-1] =  dLdH[-1].T @ np.diag( 1 - np.power(tanh(A[:,-1]),2) ) 
        # recursively compute the rest for hiddens and activations
        for t in range(T - 2, -1, -1):
            dLdH[t] = dLdout.T[t] @ self.V + dLdA[:,t+1] @ self.W
            dLdA[:,t] =  dLdH[t].T @ np.diag( 1 - np.power(tanh(A[:,t]),2) ) 

        # w.r.t W
        self.grad_W = dLdA @ H0.T
        # lastly w.r.t U and b
        self.grad_U = dLdA @ X.T
        self.grad_b = np.sum(dLdA, axis=1, keepdims=True)

        # clipping ranges
        clipmax = 5
        clipmin = -5
        # V
        self.grad_V = np.where(self.grad_V<clipmax, self.grad_V,clipmax)
        self.grad_V = np.where(self.grad_V>clipmin, self.grad_V,clipmin) 
        # bias C
        self.grad_c = np.where(self.grad_c<clipmax, self.grad_c,clipmax)
        self.grad_c = np.where(self.grad_c>clipmin, self.grad_c,clipmin)
        # W
        self.grad_W = np.where(self.grad_W<clipmax, self.grad_W,clipmax)
        self.grad_W = np.where(self.grad_W>clipmin, self.grad_W,clipmin)
        # U
        self.grad_U = np.where(self.grad_U<clipmax, self.grad_U,clipmax)
        self.grad_U = np.where(self.grad_U>clipmin, self.grad_U,clipmin)
        # bias b
        self.grad_b = np.where(self.grad_b<clipmax, self.grad_b, clipmax)
        self.grad_b = np.where(self.grad_b>clipmin, self.grad_b,clipmin)

    def update_param(self, m, grad, lr):
        # updates with adagrad
        # return: value to substract from parameter
        eps = np.finfo('float').eps
        return (lr / np.sqrt(m + eps)) * grad 

    def training(self, book_data, lr=0.1, epochs=2, print_loss=100, 
            print_generate=500, logging = False):
        N = len(book_data)
        n_sequences = math.floor(N / self.seq_max_length)

        it = 0
        losses = []
        smooth_loss = 0

        # keep track of best model
        best_loss = 100000000
        best_model = self.weigths_dict()

        # EPOCHS
        for epoch in tqdm(range(epochs)):
            print("Epoch: ", epoch)
            e = 0 # 0-indexed in python, no matlab here :)
            hprev = np.zeros((self.m,1)) # init to h0

            # ITERATING EACH SEQUENCE i
            for i in range(n_sequences):
                # extract sentence input and label
                X_chars = book_data[e : e+self.seq_max_length]
                X = np.zeros((self.K, self.seq_max_length),dtype=int)
                Y_chars = book_data[e+1 : e+self.seq_max_length+1] 
                Y = np.zeros((self.K, self.seq_max_length),dtype=int)

                # fill in one-hot encodings
                for j in range(self.seq_max_length):
                    X[:,j] = self.char_to_onehot(X_chars[j]) 
                    Y[:,j] = self.char_to_onehot(Y_chars[j]) 

                # forward pass!
                P, H, A = self.forward(X, hprev)
                # hidden for label indexes
                H0 = np.zeros((self.m, len(X_chars)))
                H0[:, [0]] = np.zeros((self.m,1))
                H0[:, 1:] = H[:, :-1]

                loss = self.compute_loss(P,Y)
                if smooth_loss == 0:
                    smooth_loss = loss
                else:
                    smooth_loss = .999 * smooth_loss + .001 * loss

                # backprop (saves grads as class variables)
                self.compute_gradients(P, X, Y, H, H0, A)

                # updates with adagrad: 
                # m_param += g*g
                # update_param(m, grad, lr)
                self.m_U += self.grad_U * self.grad_U 
                self.U -= self.update_param(self.m_U, self.grad_U, lr)

                self.m_W += self.grad_W * self.grad_W
                self.W -= self.update_param(self.m_W, self.grad_W, lr)

                self.m_V += self.grad_V * self.grad_V
                self.V -= self.update_param(self.m_V, self.grad_V, lr)

                self.m_b += self.grad_b * self.grad_b
                self.b -= self.update_param(self.m_b, self.grad_b, lr)

                self.m_c += self.grad_c * self.grad_c
                self.c -= self.update_param(self.m_c, self.grad_c, lr)

                hprev = H[:, -1]
                hprev = H[:,-1].reshape(self.m,1)

                # save and print loss info, and check best model
                if it % print_loss == 0:
                    losses.append(smooth_loss)
                    print("Epoch {}, Iteration {}, s.loss {}".format(epoch, it, smooth_loss))
                    if smooth_loss < best_loss:
                        best_loss = smooth_loss
                        best_model = self.weigths_dict()

                # print generation
                if it % print_generate == 0:
                    n_generate = 200
                    x0 = X[:,0].reshape(self.K,1)
                    Y_gen = self.generate(x0, hprev, n_generate)

                    # build sentence string
                    gen_sentence = ""
                    for j in range(Y_gen.shape[1]):
                        y = Y_gen[:,j].reshape(self.K,1)
                        gen_sentence += self.onehot_to_char(y.T)
                    s = "Epoch {}, Iteration {}, loss {}: \n{}\n".format(epoch, it, smooth_loss, gen_sentence)
                    print(s)
                    if logging:
                        write_log(s)

                it += 1
                #if it >= 10000:
                #    break
                e += self.seq_max_length

        return losses,best_loss, best_model

# ------- END CLASS

def write_log(line, filename='logs.txt'):
    with open(filename, 'a') as f:
        f.write(line + "\n")

def main():
    np.random.seed(400)
    filename = "goblet_book.txt"
    book_data, char2idx, idx2char = get_data(filename)
    K = len(char2idx.keys())
    #print("char2idx: ", char2idx)
    print("idx2char: ", idx2char)
    print("Number of datapoints: ", len(book_data))
    print("Number of chars: ", K)

    print("Initialising model...")
    m = 100
    seq_max_length = 25
    net = RNN(m, K, char2idx, idx2char,seq_max_length)

    lr = 0.1
    epochs = 4
    #epochs = 2

    print("Training beginning...")
    losses, best_loss, best_weights = net.training(
        book_data = book_data,
        lr = lr,
        epochs = epochs,
        print_loss = 100,
        #print_generate = 10000, 
        print_generate = 5000, 
        #print_loss = 100,
        #print_generate = 500,
        logging = False 
    )
    # save best weights
    write_log(str(best_weights), filename='model.txt')

    print("DONE!!!")
    print("Best loss: ", best_loss)

    # load best model and generate 
    net.load_weigths(best_weights)
      
    # generate with best model
    n_generate = 1000 
    gen_sentence = ""
    x0 = net.char_to_onehot('.').T
    h0 = np.zeros((net.m,1))
    Y_gen = net.generate(x0, h0, n_generate)

    for j in range(Y_gen.shape[1]):
        y = Y_gen[:,j].reshape(net.K,1)
        gen_sentence += net.onehot_to_char(y.T)
    print("SENTENCE 1000 CHARS:\n\n ", gen_sentence)

    # plot the loss
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel('iterations / 100')
    ax.set_ylabel('loss')
    ax.plot(losses)
    plt.show()
    
if __name__ == "__main__":
    main()
