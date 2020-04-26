import numpy as np
import random
from pypinyin import pinyin, lazy_pinyin, Style
import re

def create_word_phonetics_vector(word, shengmu_to_index,yunmu_to_index,vocab_size,char_to_ix):

    '''
    对于单个汉字，生成属于他的representation vector
    input：
    word  -- 单个汉字
    shengmu_to_index -- 声母对下标的哈希表
    yunmu_to_index -- 韵母对下标的哈希表
    vocab_size -- 一共有多少汉字
    
    
    output：
    vector -- 代表这个汉字的向量 其dimension 为 （vocab_size+extra_length,1 ）
    '''
    extra_length = len(shengmu_to_index)+len(yunmu_to_index)+4  # 除了one-hot部分的长度

    vector = np.zeros([vocab_size+extra_length,1])
    
    if word == None: return vector    #如果输入的是none，直接输出全是0的vector
    
    s,y,t = serperate_pinyin(word) #声母, 韵母， 音调提取
    s_index,y_index,t_index = get_index(s,y,t,shengmu_to_index,yunmu_to_index)
    w = char_to_ix[word]

    if s_index:    #代表有声母的情况
        vector[s_index]=1
    if y_index:    #代表有韵母的情况
        vector[y_index]=1
    if t_index:    #代表有声调的情况
        vector[t_index]=1

    vector[w] = 1  # 代表one-hot 的部分
   
    return vector

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001

def print_sample(sample_ix, ix_to_char):
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]  # capitalize first character 
    print ('%s' % (txt, ), end='')

def get_initial_loss(vocab_size, seq_length):
    return -np.log(1.0/vocab_size)*seq_length

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def initialize_parameters(n_a, n_x, n_y):
    """
    Initialize parameters with small random values
    
    Returns:
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    """
    np.random.seed(1)
    Wax = np.random.randn(n_a, n_x)*0.01 # input to hidden
    Waa = np.random.randn(n_a, n_a)*0.01 # hidden to hidden
    Wya = np.random.randn(n_y, n_a)*0.01 # hidden to output
    b = np.zeros((n_a, 1)) # hidden bias
    by = np.zeros((n_y, 1)) # output bias
    
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b,"by": by}
    
    return parameters

def rnn_step_forward(parameters, a_prev, x):
    
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    a_next = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b) # hidden state
    p_t = softmax(np.dot(Wya, a_next) + by) # unnormalized log probabilities for next chars # probabilities for next chars 
    
    return a_next, p_t

def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):
    
    gradients['dWya'] += np.dot(dy, a.T)
    gradients['dby'] += dy
    da = np.dot(parameters['Wya'].T, dy) + gradients['da_next'] # backprop into h
    daraw = (1 - a * a) * da # backprop through tanh nonlinearity
    gradients['db'] += daraw
    gradients['dWax'] += np.dot(daraw, x.T)
    gradients['dWaa'] += np.dot(daraw, a_prev.T)
    gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)
    return gradients

def update_parameters(parameters, gradients, lr):

    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Wya'] += -lr * gradients['dWya']
    parameters['b']  += -lr * gradients['db']
    parameters['by']  += -lr * gradients['dby']
    return parameters

def rnn_forward(X, Y, a0, parameters, vocab_size,shengmu_to_index,yunmu_to_index,ix_to_char,char_to_ix):
    
    extra =len(shengmu_to_index)+len(yunmu_to_index)+4
    # Initialize x, a and y_hat as empty dictionaries
    x, a, y_hat = {}, {}, {}
    
    a[-1] = np.copy(a0)
    
    # initialize your loss to 0
    loss = 0
    
    for t in range(len(X)):
        
        # Set x[t] to be the one-hot vector representation of the t'th character in X.
        # if X[t] == None, we just have x[t]=0. This is used to set the input for the first timestep to the zero vector. 
        
        if not X[t]:
            word = None
        else:
            word = ix_to_char[X[t]]
        x[t] = create_word_phonetics_vector(word, shengmu_to_index,yunmu_to_index,vocab_size,char_to_ix)
        
        # Run one step forward of the RNN
        a[t], y_hat[t] = rnn_step_forward(parameters, a[t-1], x[t])
        
        # Update the loss by substracting the cross-entropy term of this time-step from it.
        loss -= np.log(y_hat[t][Y[t]-extra,0])
        
    cache = (y_hat, a, x)
        
    return loss, cache

def rnn_backward(X, Y, parameters, cache,extra):
    # Initialize gradients as an empty dictionary
    gradients = {}
    
    # Retrieve from cache and parameters
    (y_hat, a, x) = cache
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    
    # each one should be initialized to zeros of the same dimension as its corresponding parameter
    gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
    gradients['db'], gradients['dby'] = np.zeros_like(b), np.zeros_like(by)
    gradients['da_next'] = np.zeros_like(a[0])
    
    ### START CODE HERE ###
    # Backpropagate through time
    for t in reversed(range(len(X))):
        dy = np.copy(y_hat[t])
        dy[Y[t]-extra] -= 1
        gradients = rnn_step_backward(dy, gradients, parameters, x[t], a[t], a[t-1])
    ### END CODE HERE ###
    
    return gradients, a


def clip(gradients, maxValue):
    '''
    Clips the gradients' values between minimum and maximum.
    
    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue
    
    Returns: 
    gradients -- a dictionary with the clipped gradients.
    '''
    
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
   
    ### START CODE HERE ###
    # clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby]. (≈2 lines)
    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out=gradient)
    ### END CODE HERE ###
    
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    
    return gradients



def sample(parameters, char_to_ix, seed,extra,ix_to_char,shengmu_to_index,yunmu_to_index):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    Arguments:
    parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b. 
    char_to_ix -- python dictionary mapping each character to an index.
    seed -- used for grading purposes. Do not worry about it.

    Returns:
    indices -- a list of length n containing the indices of the sampled characters.
    """
    
    # Retrieve parameters and relevant shapes from "parameters" dictionary
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    
    ### START CODE HERE ###
    # Step 1: Create the a zero vector x that can be used as the one-hot vector 
    # representing the first character (initializing the sequence generation). (≈1 line)
    x = np.zeros([vocab_size+extra,1])
    # Step 1': Initialize a_prev as zeros (≈1 line)
    a_prev = np.zeros([n_a,1])
    
    # Create an empty list of indices, this is the list which will contain the list of indices of the characters to generate (≈1 line)
    indices = []
    
    # idx is the index of the one-hot vector x that is set to 1
    # All other positions in x are zero.
    # We will initialize idx to -1
    idx = -1 
    
    # Loop over time-steps t. At each time-step:
    # sample a character from a probability distribution 
    # and append its index (`idx`) to the list "indices". 
    # We'll stop if we reach 50 characters 
    # (which should be very unlikely with a well trained model).
    # Setting the maximum number of characters helps with debugging and prevents infinite loops. 
    counter = 0
    newline_character = char_to_ix['\n']
    
    while (idx != newline_character and counter != 3):
        
        # Step 2: Forward propagate x using the equations (1), (2) and (3)
        a = np.tanh(np.dot(Waa,a_prev)+np.dot(Wax,x)+b)
        z = np.dot(Wya,a)+by
        y = softmax(z)
        
        # for grading purposes
        np.random.seed(counter+seed) 
        
        # Step 3: Sample the index of a character within the vocabulary from the probability distribution y
        # (see additional hints above)
        idx = np.random.choice([i for i in range(vocab_size)],p = y.ravel()) +extra

        # Append the index to "indices"
        indices.append(idx)
        
        # Step 4: Overwrite the input x with one that corresponds to the sampled index `idx`.
        # (see additional hints above)
        word = ix_to_char[idx]
        x = create_word_phonetics_vector(word, shengmu_to_index,yunmu_to_index,vocab_size,char_to_ix)
        
        
        # Update "a_prev" to be "a"
        a_prev = a
        
        # for grading purposes
        seed += 1
        counter +=1
        
    ### END CODE HERE ###

    if (counter == 3):
        indices.append(char_to_ix['\n'])
    
    return indices


def optimize(X, Y, a_prev, parameters, learning_rate,shengmu_to_index,yunmu_to_index,ix_to_char,char_to_ix):
    """
    Execute one step of the optimization to train the model.
    
    Arguments:
    X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
    Y -- list of integers, exactly the same as X but shifted one index to the left.
    a_prev -- previous hidden state.
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    learning_rate -- learning rate for the model.
    
    Returns:
    loss -- value of the loss function (cross-entropy)
    gradients -- python dictionary containing:
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                        db -- Gradients of bias vector, of shape (n_a, 1)
                        dby -- Gradients of output bias vector, of shape (n_y, 1)
    a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
    """
    
    ### START CODE HERE ###
    vocab_size = parameters['Wya'].shape[0]
    extra =len(shengmu_to_index)+len(yunmu_to_index)+4
    # Forward propagate through time (≈1 line)
    loss, cache = rnn_forward(X, Y, a_prev, parameters,vocab_size,shengmu_to_index,yunmu_to_index,ix_to_char,char_to_ix )
    
    # Backpropagate through time (≈1 line)
    gradients, a = rnn_backward(X, Y, parameters, cache,extra)
    
    # Clip your gradients between -5 (min) and 5 (max) (≈1 line)
    gradients = clip(gradients, 5)
    
    # Update parameters (≈1 line)
    parameters = update_parameters(parameters, gradients, learning_rate)
    
    ### END CODE HERE ###
    
    return loss, gradients, a[len(X)-1]


def model(data, ix_to_char, char_to_ix, num_iterations, n_a,dino_names,vocab_size,extra_length,shengmu_to_index,yunmu_to_index):
    """
    Trains the model and generates dinosaur names. 
    
    Arguments:
    data -- text corpus
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number of iterations to train the model for
    n_a -- number of units of the RNN cell
    dino_names -- number of dinosaur names you want to sample at each iteration. 
    vocab_size -- number of unique characters found in the text (size of the vocabulary)
    
    Returns:
    parameters -- learned parameters
    """
    
    # Retrieve n_x and n_y from vocab_size
    n_x, n_y = vocab_size+extra_length, vocab_size
    
    # Initialize parameters
    parameters = initialize_parameters(n_a, n_x, n_y)
    
    # Initialize loss (this is required because we want to smooth our loss)
    loss = get_initial_loss(vocab_size, dino_names)
    
    # Build list of all dinosaur names (training examples).
    examples = data
    
    # Shuffle list of all dinosaur names
    np.random.seed(0)
    np.random.shuffle(examples)
    
    # Initialize the hidden state of your LSTM
    a_prev = np.zeros((n_a, 1))
    
    # Optimization loop
    for j in range(num_iterations):
        
        ### START CODE HERE ###
        
        # Set the index `idx` (see instructions above)
        idx = j % len(examples)
        
        # Set the input X (see instructions above)
        single_example = examples[idx]
        single_example_chars = [c for c in single_example]
        single_example_ix = [char_to_ix[c] for c in single_example_chars]
        X = [None]+single_example_ix
        
        # Set the labels Y (see instructions above)
        ix_newline = '\n'
        Y = X[1:]+[char_to_ix[ix_newline]]
        
        # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
        # Choose a learning rate of 0.01
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, 0.01,shengmu_to_index,yunmu_to_index,ix_to_char,char_to_ix)
        
        ### END CODE HERE ###
        
        # Use a latency trick to keep the loss smooth. It happens here to accelerate the training.
        loss = smooth(loss, curr_loss)

        # Every 2000 Iteration, generate "n" characters thanks to sample() to check if the model is learning properly
        if j % 2000 == 0:
            
            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
            
            # The number of dinosaur names to print
            seed = 0
            for name in range(dino_names):
                
                # Sample indices and print them
                sampled_indices = sample(parameters, char_to_ix, seed,extra_length,ix_to_char,shengmu_to_index,yunmu_to_index)
                print_sample(sampled_indices, ix_to_char)
                
                seed += 1  # To get the same result (for grading purposes), increment the seed by one. 
      
            print('\n')
        
    return parameters


from pypinyin import pinyin, lazy_pinyin, Style
import numpy as np
import re


def serperate_pinyin(w):
    '''
    将输入的单个汉字声母，韵母，声调提取出来
    
    input:
    w -- 输入的汉字，必须是单个，如果是多个汉字只会匹配第一个汉字
    
    ouput:
    shengmu -- 该汉字的声母，如果该汉字没有声母，则输出空字符串
    yunmu --  该汉字的韵母，没有声调标注
    tone -- 该汉字的声调，如果是亲生则声调为0
    '''
    #检测输入的是否是中文
    is_chinese = re.compile('[\u4e00-\u9fa5]')
    if not is_chinese.findall(w):
        return '', '',0
    
    #提取声母
    shengmu_list = pinyin(w,style=Style.INITIALS,strict = False) #不然 y， w不会算在里面
    shengmu = shengmu_list[0][0]
    
    
    #提取韵母
    yunmu_list= pinyin(w,style=Style.FINALS,strict = True)   #strict =  True 主要是为了区分 un和vn的区别
    yunmu = yunmu_list[0][0]
    
    #提取声调
    pre_tone = pinyin(w, style=Style.FINALS_TONE3)            #先把韵母和声调一起提取出来 比如 中 => ong1
    pattern=re.compile('[1-4]')                              #用正则表达式匹配其中的数字
    tone_list = pattern.findall(pre_tone[0][0]) 
    if tone_list:
        tone = int(tone_list[0])                                 #如果是轻声则检测不到声调
    else:
        tone = 0              
    return shengmu,yunmu,tone


def get_index(s,y,t,shengmu_to_index,yunmu_to_index):
    '''
    获取声母s，韵母y，音调t在向量中的下标位
    input:
        s -- 表示输入的声母
        y -- 表示输入的韵母
        t -- 表示输入的音调

    output:
        s_index -- 声母在向量中的下标
        y_index -- 韵母在向量中的下标
        t_index -- 音调在向量中的下标
    '''
    if not s: # 如果没有声母的情况
        s_index =None
    else:
        s_index = shengmu_to_index[s]
    
    if not y : #如果没有韵母的情况
        y_index = None
    else:
        y_index = yunmu_to_index[y]
    
    
    if t ==0:  #没有音调的情况
        t_index  =  None
    else:
        t_index = t+len(shengmu_to_index)+len(yunmu_to_index)-1    
    
    return s_index,y_index,t_index





def extract_names():
    '''
    从名字列表里正则匹配出中文名字,剔除重复的名字
    output:
    names -- 含有所有名字的列表
    '''
    pattern = re.compile('[\u4e00-\u9fa5]+')   
    names = []
    with open('names.txt','r') as f:
        for line in f.readlines(): 
            name = pattern.findall(line)
            if name: names.append(name[0])
    return list(set(names))

def extract_word():
    '''
    将所有组成名字的字都给列出来
    output:
    words -- 所有包含字组成的列表
    
    '''
    data = open('names.txt','r').read()
    data= list(set(data))
    pattern = re.compile('[\u4e00-\u9fa5]')
    words = [pattern.findall(line)[0] for line in data if pattern.findall(line)]
    return words





