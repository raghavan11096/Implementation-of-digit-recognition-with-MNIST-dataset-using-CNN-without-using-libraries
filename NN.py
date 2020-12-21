################################################################################
#
# LOGISTICS
#
#    RAGHAVAN SIVAKUMAR
#    RXS190048
#
# FILE
#
#    <nn.py>
#
# DESCRIPTION
#
#    MNIST image classification with an xNN written and trained in Python
#
# INSTRUCTIONS
#
#    1. Go to Google Colaboratory: https://colab.research.google.com/notebooks/welcome.ipynb
#    2. File - New Python 3 notebook
#    3. Cut and paste this file into the cell (feel free to divide into multiple cells)
#    4. Runtime - Run all
#
# NOTES
#
#    1. A summary of my nn.py code:
#
#       <Forward path code summary / highlights>
  
  #       There are three layers:- Initial_layer, output_layer, and hidden layer, where the input layer is the initial layer that has the input weights which is multiplied with hidden layer values and passed to activation function __Relu
#	the output obtained is passed to the output layer, where the softmax function is called to find the probablity and the crossentropy function is called to perform error loss.
         
#       <Error code summary / highlights>
#	The function compute_loss performs the error loss which is obtained from forward function, that calls crossentropy function that calculates the error

#       <Backward path code summary / highlights>
 #       The function backward_propogation is called which helps in propogating backwards, by multiplying with the gradient with the values, thus weight_update is called to update the weights.

#       <Weight update code summary / highlights>
#	The weight_update is performed after the back propogation, for every iteration, thus keeping account of accuracy

#
#    2. Accuracy display
#
#       <Per epoch display info cut and pasted from your training output>
#       <Final accuracy cut and pasted from your training output>
# No of items:1000
# No of items:2000
# No of items:3000
# No of items:4000
# Accuarcy   88.67
# loss 23278.36108436058 test 0.8867
# Epochs:1
# No of items:6000
# No of items:7000
# No of items:8000
# No of items:9000
# Accuarcy   90.6
# loss 19792.40651611106 test 0.906
# Epochs:2
# No of items:11000
# No of items:12000
# No of items:13000
# No of items:14000
# Accuarcy   92.11
# loss 16639.30908234097 test 0.9211
# Epochs:3
# No of items:16000
# No of items:17000
# No of items:18000
# No of items:19000
# Accuarcy   93.33
# loss 13710.087485005803 test 0.9333
# Epochs:4
# No of items:21000
# No of items:22000
# No of items:23000
# No of items:24000
# Accuarcy   94.44
# loss 11568.216526733151 test 0.9444
# Epochs:5
# No of items:26000
# No of items:27000
# No of items:28000
# No of items:29000
# Accuarcy   94.74
# loss 10637.497924092071 test 0.9474
# Epochs:6
# No of items:31000
# No of items:32000
# No of items:33000
# No of items:34000
# Accuarcy   95.02
# loss 10092.73689086701 test 0.9502
# Epochs:7
# No of items:36000
# No of items:37000
# No of items:38000
# No of items:39000
# Accuarcy   95.38
# loss 8920.112269214544 test 0.9538
# Epochs:8
# No of items:41000
# No of items:42000
# No of items:43000
# No of items:44000
# Accuarcy   95.78
# loss 8420.320370117799 test 0.9578
# Epochs:9
# No of items:46000
# No of items:47000
# No of items:48000
# No of items:49000
# Accuarcy   95.89
# loss 7960.685682415735 test 0.9589
# Epochs:10
# No of items:51000
# No of items:52000
# No of items:53000
# No of items:54000
# Accuarcy   95.38
# loss 8602.457520714903 test 0.9538
# Epochs:11
# No of items:56000
# No of items:57000
# No of items:58000
# No of items:59000
# Accuarcy   95.76
# loss 7760.994666915873 test 0.9576
# Epochs:12
# No of items:61000
# No of items:62000
# No of items:63000
# No of items:64000
# Accuarcy   95.06
# loss 8779.318519769755 test 0.9506
# Epochs:13
# No of items:66000
# No of items:67000
# No of items:68000
# No of items:69000
# Accuarcy   96.01
# loss 7082.019248243393 test 0.9601
# Epochs:14
# No of items:71000
# No of items:72000
# No of items:73000
# No of items:74000
# Accuarcy   95.87
# loss 7455.179067415745 test 0.9587
# Epochs:15
# No of items:76000
# No of items:77000
# No of items:78000
# No of items:79000
# Accuarcy   96.47
# loss 5841.366221866519 test 0.9647
# Epochs:16
# No of items:81000
# No of items:82000
# No of items:83000
# No of items:84000
# Accuarcy   96.25
# loss 6686.312755861731 test 0.9625
# Epochs:17
# No of items:86000
# No of items:87000
# No of items:88000
# No of items:89000
# Accuarcy   95.96
# loss 6980.993041983519 test 0.9596
# Epochs:18
# No of items:91000
# No of items:92000
# No of items:93000
# No of items:94000
# Accuarcy   96.18
# loss 6388.592526946037 test 0.9618
# Epochs:19
# No of items:96000
# No of items:97000
# No of items:98000
# No of items:99000
# Accuarcy   96.74
# loss 5478.675403815366 test 0.9674
# Epochs:20
# No of items:101000
# No of items:102000
# No of items:103000
# No of items:104000
# Accuarcy   96.59
# loss 5858.868411225458 test 0.9659
# Epochs:21
# No of items:106000
# No of items:107000
# No of items:108000
# No of items:109000
# Accuarcy   96.92
# loss 5094.190356112698 test 0.9692
# Epochs:22
# No of items:111000
# No of items:112000
# No of items:113000
# No of items:114000
# Accuarcy   97.2
# loss 4692.545750126634 test 0.972
# Epochs:23
# No of items:116000
# No of items:117000
# No of items:118000
# No of items:119000
# Accuarcy   96.85
# loss 4886.070358048491 test 0.9685
# Epochs:24
# No of items:121000
# No of items:122000
# No of items:123000
# No of items:124000
# Accuarcy   97.2
# loss 4698.606673518741 test 0.972
# Epochs:25
# No of items:126000
# No of items:127000
# No of items:128000
# No of items:129000
# Accuarcy   96.93
# loss 4867.288296681773 test 0.9693
# Epochs:26
# No of items:131000
# No of items:132000
# No of items:133000
# No of items:134000
# Accuarcy   96.95
# loss 5048.54984241862 test 0.9695
# Epochs:27
# No of items:136000
# No of items:137000
# No of items:138000
# No of items:139000
# Accuarcy   96.1
# loss 5893.487907256995 test 0.961
# Epochs:28
# No of items:141000
# No of items:142000
# No of items:143000
# No of items:144000
# Accuarcy   96.85
# loss 4560.5529980100055 test 0.9685
# Epochs:29
# No of items:146000
# No of items:147000
# No of items:148000
# No of items:149000
# Accuarcy   96.92
# loss 4461.3475150222985 test 0.9692
# Epochs:30
# No of items:151000
# No of items:152000
# No of items:153000
# No of items:154000
# Accuarcy   97.26
# loss 4248.100070392827 test 0.9726
# Epochs:31
# No of items:156000
# No of items:157000
# No of items:158000
# No of items:159000
# Accuarcy   97.13
# loss 4257.951316429783 test 0.9713
# Epochs:32
# No of items:161000
# No of items:162000
# No of items:163000
# No of items:164000
# Accuarcy   97.36
# loss 3822.0660048123627 test 0.9736
# Epochs:33
# No of items:166000
# No of items:167000
# No of items:168000
# No of items:169000
# Accuarcy   97.46
# loss 3624.7098058451365 test 0.9746
# Epochs:34
# No of items:171000
# No of items:172000
# No of items:173000
# No of items:174000
# Accuarcy   97.37
# loss 3772.803705557243 test 0.9737
# Epochs:35
# No of items:176000
# No of items:177000
# No of items:178000
# No of items:179000
# Accuarcy   97.49
# loss 3622.7091489203212 test 0.9749
# Epochs:36
# No of items:181000
# No of items:182000
# No of items:183000
# No of items:184000
# Accuarcy   97.4
# loss 3387.43429714986 test 0.974
# Epochs:37
# No of items:186000
# No of items:187000
# No of items:188000
# No of items:189000
# Accuarcy   97.32
# loss 3570.836688842431 test 0.9732
# Epochs:38
# No of items:191000
# No of items:192000
# No of items:193000
# No of items:194000
# Accuarcy   97.38
# loss 3358.5891742241865 test 0.9738
# Epochs:39
# No of items:196000
# No of items:197000
# No of items:198000
# No of items:199000
# Accuarcy   97.64
# loss 3196.867543027347 test 0.9764
# Epochs:40
# Accuarcy   97.64

#
################################################################################

import os.path
import urllib.request
import gzip
import math
import numpy             as np
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
from random import randint

################################################################################
#
# PARAMETERS
#
################################################################################

#
# add other hyper parameters here with some logical organization
#

# data
DATA_NUM_TRAIN         = 60000
DATA_NUM_TEST          = 10000
DATA_CHANNELS          = 1
DATA_ROWS              = 28
DATA_COLS              = 28
DATA_CLASSES           = 10
DATA_URL_TRAIN_DATA    = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
DATA_URL_TRAIN_LABELS  = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
DATA_URL_TEST_DATA     = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
DATA_URL_TEST_LABELS   = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
DATA_FILE_TRAIN_DATA   = 'train_data.gz'
DATA_FILE_TRAIN_LABELS = 'train_labels.gz'
DATA_FILE_TEST_DATA    = 'test_data.gz'
DATA_FILE_TEST_LABELS  = 'test_labels.gz'

# display
DISPLAY_ROWS   = 8
DISPLAY_COLS   = 4
DISPLAY_COL_IN = 10
DISPLAY_ROW_IN = 25
DISPLAY_NUM    = DISPLAY_ROWS*DISPLAY_COLS

################################################################################
#
# DATA
#
################################################################################

# download
if (os.path.exists(DATA_FILE_TRAIN_DATA)   == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_DATA,   DATA_FILE_TRAIN_DATA)
if (os.path.exists(DATA_FILE_TRAIN_LABELS) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_LABELS, DATA_FILE_TRAIN_LABELS)
if (os.path.exists(DATA_FILE_TEST_DATA)    == False):
    urllib.request.urlretrieve(DATA_URL_TEST_DATA,    DATA_FILE_TEST_DATA)
if (os.path.exists(DATA_FILE_TEST_LABELS)  == False):
    urllib.request.urlretrieve(DATA_URL_TEST_LABELS,  DATA_FILE_TEST_LABELS)

# training data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_train_data   = gzip.open(DATA_FILE_TRAIN_DATA, 'r')
file_train_data.read(16)
buffer_train_data = file_train_data.read(DATA_NUM_TRAIN*DATA_ROWS*DATA_COLS)
train_data        = np.frombuffer(buffer_train_data, dtype=np.uint8).astype(np.float32)
train_data        = train_data.reshape(DATA_NUM_TRAIN, 1, DATA_ROWS, DATA_COLS)

# training labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_train_labels   = gzip.open(DATA_FILE_TRAIN_LABELS, 'r')
file_train_labels.read(8)
buffer_train_labels = file_train_labels.read(DATA_NUM_TRAIN)
train_labels        = np.frombuffer(buffer_train_labels, dtype=np.uint8).astype(np.int32)

# testing data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_test_data   = gzip.open(DATA_FILE_TEST_DATA, 'r')
file_test_data.read(16)
buffer_test_data = file_test_data.read(DATA_NUM_TEST*DATA_ROWS*DATA_COLS)
test_data        = np.frombuffer(buffer_test_data, dtype=np.uint8).astype(np.float32)
test_data        = test_data.reshape(DATA_NUM_TEST, 1, DATA_ROWS, DATA_COLS)

# testing labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_test_labels   = gzip.open(DATA_FILE_TEST_LABELS, 'r')
file_test_labels.read(8)
buffer_test_labels = file_test_labels.read(DATA_NUM_TEST)
test_labels        = np.frombuffer(buffer_test_labels, dtype=np.uint8).astype(np.int32)
fig = plt.figure(figsize=(DISPLAY_COL_IN, DISPLAY_ROW_IN))
ax  = []
for i in range(DISPLAY_NUM):
    img = test_data[i, :, :, :].reshape((DATA_ROWS, DATA_COLS))
    ax.append(fig.add_subplot(DISPLAY_ROWS, DISPLAY_COLS, i + 1))
    ax[-1].set_title('True: ' + str(test_labels[i]) + ' xNN: ' + str(test_labels[i]))
    plt.imshow(img, cmap='Greys')
plt.show()

# Reshaping the training data
train_data=train_data.reshape(60000,784)
print(train_data.shape)
#reshaping the testing data
test_data=test_data.reshape(10000,784)
print(test_data.shape)
#dividing the train and test data by number of pixels
train_data,test_data=train_data/255,test_data/255

X_train,X_test=train_data,test_data
Y_train,Y_test=train_labels,test_labels
print(train_labels.shape,test_labels.shape)

class NeuralNet:
    input_values = {}
    output_values = {}

    def __init__(self, inputs, hidden, outputs):
        # layers initialization
        self.input_values['para'] = np.random.randn(hidden, inputs) / np.sqrt(num_inputs)
        self.input_values['bias'] = np.random.randn(hidden, 1) / np.sqrt(hidden)
        self.output_values['para'] = np.random.randn(outputs, hidden) / np.sqrt(hidden)
        self.output_values['bias'] = np.random.randn(outputs, 1) / np.sqrt(hidden)

        self.input_size = inputs
        self.hid_size = hidden
        self.output_size = outputs

    def __relu(self, initial_layer, type='ReLU', deri=False):
        # implement the activation function
        if type == 'ReLU':
            if deri == True:
                return np.array([1 if i > 0 else 0 for i in np.squeeze(initial_layer)])
            else:
                return np.array([i if i > 0 else 0 for i in np.squeeze(initial_layer)])
        elif type == 'Sigmoid':
            if deri == True:
                return 1 / (1 + np.exp(-initial_layer)) * (1 - 1 / (1 + np.exp(-initial_layer)))
            else:
                return 1 / (1 + np.exp(-initial_layer))
        elif type == 'tanh':
            if deri == True:
                return
            else:
                return 1 - (np.tanh(initial_layer)) ** 2
        else:
            raise TypeError('Invalid type!')

    def softmax(self, z):
        # calculate softmax
        return 1 / sum(np.exp(z)) * np.exp(z)

    def crossentropy(self, p, y):
        # calculate cross entropy
        return -np.log(p[y])

    def forward(self, x, y):
        # Forward propogation
        initial_layer = np.matmul(self.input_values['para'], x).reshape((self.hid_size, 1)) + self.input_values['bias']
        hidden_layer = np.array(self.__relu(initial_layer)).reshape((self.hid_size, 1))
        final_layer = np.matmul(self.output_values['para'], hidden_layer).reshape((self.output_size, 1)) + \
                      self.output_values['bias']
        predict_list = np.squeeze(self.softmax(final_layer))
        error = self.crossentropy(predict_list, y)

        network = {
            'initial_layer': initial_layer,
            'hidden_layer': hidden_layer,
            'final_layer': final_layer,
            'f(X)': predict_list.reshape((1, self.output_size)),
            'error': error
        }
        return network

    def backpropogate(self, x, y, f_result):
        # Error calculation
        E = np.array([0] * self.output_size).reshape((1, self.output_size))
        E[0][y] = 1
        # Passing the output gradient
        output_layer_derivative = (-(E - f_result['f(X)'])).reshape((self.output_size, 1))
        db_2 = copy.copy(output_layer_derivative)
        dC = np.matmul(output_layer_derivative, f_result['hidden_layer'].transpose())
        delta = np.matmul(self.output_values['para'].transpose(), output_layer_derivative)
        db_1 = delta.reshape(self.hid_size, 1) * self.__relu(f_result['initial_layer'], deri=True).reshape(
            self.hid_size, 1)
        dW = np.matmul(db_1.reshape((self.hid_size, 1)), x.reshape((1, 784)))

        grad = {
            'dC': dC,
            'db_2': db_2,
            'db_1': db_1,
            'dW': dW
        }
        return grad

    def weightUpdate(self, b_result, learning_rate):
        # Weight update
        self.output_values['para'] -= learning_rate * b_result['dC']
        self.output_values['bias'] -= learning_rate * b_result['db_2']
        self.input_values['bias'] -= learning_rate * b_result['db_1']
        self.input_values['para'] -= learning_rate * b_result['dW']

    def computeLoss(self, X_train, Y_train):
        # implement the loss function of the training set
        loss = 0
        for n in range(len(X_train)):
            y = Y_train[n]
            x = X_train[n][:]
            loss += self.forward(x, y)['error']
        return loss

    def train(self, X_train, Y_train, training_batch=1000, learning_rate=0.5):
        # random values generation for training
        indexes = np.random.choice(len(X_train), training_batch, replace=True)

        def l_rate(base_rate, ite, training_batch, interval=False):
            # learning rate
            if interval == True:
                return base_rate * 10 ** (-np.floor(ite / training_batch * 5))
            else:
                return base_rate

        trained = 1
        loss_vals = {}
        test_vals = {}
        epochs = 0
        for i in indexes:
            f_result = self.forward(X_train[i], Y_train[i])
            b_result = self.backpropogate(X_train[i], Y_train[i], f_result)
            self.weightUpdate(b_result, l_rate(learning_rate, i, training_batch, True))

            if trained % 1000 == 0:
                if trained % 5000 == 0:
                    epochs += 1
                    loss = self.computeLoss(X_train, Y_train)
                    test = self.testing(X_test, Y_test)
                    print("loss", loss, "test", test)
                    loss_vals[str(trained)] = loss
                    test_vals[str(trained)] = test
                    print('Epochs:{}'.format(epochs))
                else:
                    print('No of items:{}'.format(trained))
            trained += 1

        return loss_vals, test_vals

    def testing(self, X_test, Y_test):
        # testing on test data
        valid = 0
        for n in range(len(X_test)):
            y = Y_test[n]
            x = X_test[n][:]
            prediction = np.argmax(self.forward(x, y)['f(X)'])
            if (prediction == y):
                valid += 1
        print('Accuarcy  ', valid * 100 / len(X_test))
        return valid / np.float(len(X_test))


# batch size initialization
training_batch = 200000
# learning rate
learning_rate = 0.01
# input size
num_inputs = 784
# digits size
num_outputs = 10
# hidden layer size
hidden_size = 300

# evaluation
model = NeuralNet(num_inputs, hidden_size, num_outputs)
lossGraph, testGraph = model.train(X_train, Y_train, training_batch=training_batch, learning_rate=learning_rate)
accuracy = model.testing(X_test, Y_test)

import matplotlib.pyplot as plt
# Batch size vs Loss Function Graph
plt.plot(*zip(*sorted(lossGraph.items())))
plt.ylabel('Loss function vs Batch Size')
plt.xlabel('Batch')
plt.title('Loss function vs Batch Size')
plt.show()
# Batch size vs Test Accuracy Function Graph
plt.plot(t*zip(*sorted(testGraph.items())))
plt.ylabel('Test Accuracy')
plt.xlabel('Batch')
plt.title('Test accuracy vs Batch Size')
plt.show()