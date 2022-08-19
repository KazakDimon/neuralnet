# Вспомогательная программа для тестирования результатов работы НС

# Возможности:
# - печать весов НС для CoDeSys
# - только тестирование существующих НС

# scipy.special for the sigmoid function expit()
import scipy.special
# Библиотека для работы с массивами данных
import numpy
# Библиотеки для работы с файлами
import os
import pickle
import errno
# Импортируем time для подсчёта времени тренировки в каждой эпохе
import time
# Библиотека для графиков
from matplotlib import cm
import matplotlib.pyplot as plt

import TestArrWithLine as data_array
now_time = time.asctime()
STORAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.storage'))
BACKUP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.temp' + now_time))

DIR = r'D:\Нужное\For NN'
# neural network class definition
class neuralNetwork:
    
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate,load_from=None):
        
        if load_from is not None:
            if not self.load(load_from):
                raise ValueError('Model with name `{}` not found'.format(load_from))
        else:
            # set number of nodes in each input, hidden, output layer
            self.inodes = inputnodes
            self.hnodes = hiddennodes
            self.onodes = outputnodes
            
            # link weight matrices, wih and who
            # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
            # w11 w21
            # w12 w22 etc 
            self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
            self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
    
            # learning rate
            self.lr = learningrate
            
            # счётчик совершенных эпох
            self.count_epochs = 0
            
            # лист информации
            self.list_inf = []
            
        
        self.a = 1
        # activation function is the sigmoid function
        #self.activation_function = lambda x: scipy.special.expit(x*self.a)
        self.activation_function = lambda x: numpy.tanh(x*self.a)
        pass

    
    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        #self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.who += self.lr * numpy.dot((output_errors * (1.0 - pow(final_outputs,2))), numpy.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        #self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        self.wih += self.lr * numpy.dot((hidden_errors * (1.0 - pow(hidden_outputs,2))), numpy.transpose(inputs))
        
        pass
    
    # Функция для подсчёта совершенных эпох
    def count_ep(self): 
        self.count_epochs+=1
    
    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

    # Создание информации о предыдущем проходе этой НС
    def information(self,inf_zero,inf_one,inf_two,inf_three):
        self.list_inf = [inf_zero,inf_one,inf_two,inf_three]
    
    # Получение информации о предыдущем проходе этой НС
    #(печать с помощью цикла)    
    def get_information(self):
        for inf in self.list_inf:
            print(inf)
    
    #Сохраняет обученную модель
    def save(self, key):
        try:
            os.makedirs(STORAGE_PATH)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        file_path = os.path.join(STORAGE_PATH, key)
        value = {
            'input_nodes': self.inodes,
            'hidden_nodes': self.hnodes,
            'output_nodes': self.onodes,
            'rate': self.lr,
            'w_i_h': self.wih,
            'w_h_o': self.who,
            'count_ep': self.count_epochs,
            'info': self.list_inf
        }
        with open(file_path, mode='wb') as fn:
            pickle.dump(value, fn, protocol=2)

    #Сохраняет обученную модель
    def save_backup(self, key):
        try:
            os.makedirs(BACKUP_PATH)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        file_path = os.path.join(BACKUP_PATH, key)
        value = {
            'input_nodes': self.inodes,
            'hidden_nodes': self.hnodes,
            'output_nodes': self.onodes,
            'rate': self.lr,
            'w_i_h': self.wih,
            'w_h_o': self.who,
            'count_ep': self.count_epochs,
            'info': self.list_inf
        }
        with open(file_path, mode='wb') as fn:
            pickle.dump(value, fn, protocol=2)   
            
    # Загружаем обученную модель
    def load(self, key):
        """Загружает обученную модель
        :param str key: имя модели
        """
        file_path = os.path.join(STORAGE_PATH, key)
        if os.path.isfile(file_path):
            with open(file_path, mode='rb') as fn:
                value = pickle.load(fn)
            self.inodes = value['input_nodes']
            self.hnodes = value['hidden_nodes']
            self.onodes = value['output_nodes']
            self.lr = value['rate']
            self.wih = value['w_i_h']
            self.who = value['w_h_o']
            self.count_epochs = value['count_ep']
            self.list_inf = value ['info']
            return True
        else:
            return False

# Начальные значения параметров, без них не инициализировать НС
input_nodes = 15
hidden_nodes = 9
output_nodes = 2
learning_rate = 0.0001

const_step = 10
step = 0
resolution = 0 # если 0 => веса для CoDeSys не печатаются
# Имя созданной НС(если была создана)
key = 'tan47.654+0.5+ep13'

n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate,key)
print('Saving model is download!')

input_nodes = n.inodes
hidden_nodes = n.hnodes
output_nodes = n.onodes
str_nodes = 'input_nodes = ' + str(input_nodes) + ' ' +\
            'hidden_nodes = ' + str(hidden_nodes) +' ' +\
            'output_nodes = ' + str(output_nodes)
print(str_nodes)
# Печатаем веса нейронной сети в файле для переноса на ПЛК
if resolution!=0:
    print('(*Веса первого слоя:*)')
    sloy1 = n.wih
    #Печать весов для CoDeSys
    count_wth = 1
    count_line = 1
    str_line = 'WSloy1_'+str(count_line)+': ARRAY[0..'+str(input_nodes)+'] OF REAL := 0.0,'
    for y in sloy1:
        for x in y:
            x = round(float(x),5)
            if count_wth<input_nodes:
                str_line+=str(x)+','
                count_wth+=1
            elif count_wth==input_nodes:
                str_line+=str(x)+';'
                print(str_line)
                count_wth = 1
                count_line+=1
        str_line = 'WSloy1_'+str(count_line)+': ARRAY[0..'+str(input_nodes)+'] OF REAL := 0.0,'
    
    print('(*Веса второго слоя:*)')
    sloy2 = n.who
    #Печать весов для CoDeSys
    count_wth = 1
    count_line = 1
    str_line = 'WSloy2_'+str(count_line)+': ARRAY[0..'+str(hidden_nodes)+'] OF REAL := 0.0,'
    for y in sloy2:
        for x in y:
            x = round(float(x),4)
            if count_wth<hidden_nodes:
                str_line+=str(x)+','
                count_wth+=1
            elif count_wth==hidden_nodes:
                str_line+=str(x)+';'
                print(str_line)
                count_wth = 1
                count_line+=1
        str_line = 'WSloy2_'+str(count_line)+': ARRAY[0..'+str(hidden_nodes)+'] OF REAL := 0.0,'

else:
    print('Разрешения на печать весов для CoDeSys не получено!')
flag_shuf = True    
test_data = data_array.main(input_nodes,flag_shuf)[1]
scorecard = []
numpy.random.shuffle(test_data)
for record in test_data:
    all_values = list(record)
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = numpy.asfarray(all_values[1:])
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    #print('Answer - ',inputs)
    print('Label - ',label)
    print('Correct label - ',correct_label)
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        print(all_values)
    
    if step==const_step:
        break
    else:
        step+=1

scorecard_array = numpy.asarray(scorecard)
# Вычисляем процент совпадений
perf = scorecard_array.sum() / scorecard_array.size
perf = round(float(perf),5) * 100
print('Процент = ',round(perf,1),'%')
