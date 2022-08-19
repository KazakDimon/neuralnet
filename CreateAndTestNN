#Программа для создания нейронных сетей 

#Возможности:
# - сигмоидальная или функция гипорболического тангенса для функции активации
# - сохранение нейронной сети на каждом шагу обучения
# - построение графиков обучения и графиков статистичекого разбороса (box whisker)

# Библиотека для сигмоидальной функции (раскоментировать, если нужна сигмоида)
# scipy.special for the sigmoid function expit()
import scipy.special
# Библиотека для работы с массивами данных
import numpy
# Импортируем time для подсчёта времени тренировки в каждой эпохе
import time
# Библиотеки для работы с файлами
import os
import pickle
import errno
import shutil
# Библиотека для графиков
import matplotlib.pyplot as plt
# Подпрограмма для подготовки данных
import TestArrWithLine as data_array

now_time = time.asctime()
now_time = now_time.replace(':',';')
# Указываем путь для сохранения и загрузки НС
STORAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.storage'))
BACKUP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.temp' + now_time))
DIR = r'E:\Нужное\For NN'

# Определение класса нейронной сети
class neuralNetwork:
    # Инициализация нейронной сети
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, alfa_const, load_from=None):
        # проверяем верно ли указана модель для загрузки НС
        # если переменная load_from не равна None
        if load_from is not None:
            if not self.load(load_from):
                raise ValueError('Model with name `{}` not found'.format(load_from))
        else:
            # устанавливаем количество входных, скрытых и выходных нейронов
            self.inodes = inputnodes
            self.hnodes = hiddennodes
            self.onodes = outputnodes
            # инициализируем веса нейронной сети
            self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
            self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))            
            # инициализируем коэффициент обучения
            self.lr = learningrate
            # инициализируем счётчик совершённых эпох
            self.count_epochs = 0
            # инициализируем лист информации
            self.list_inf = []
        
        # инициализируем константу в функции активации
        self.a = alfa_const
        # Функция активации - сигмойда
        #self.activation_function = lambda x: scipy.special.expit(x*self.a)
        # Функция активации - гиперболический тангенс
        self.activation_function = lambda x: numpy.tanh(x*self.a)
        pass
    # Метод обучения нейронной сети
    def train(self, inputs_list, targets_list): 
        # преобразуем входные значения из списка в 2-й массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # вычисляем сигнал к скрытому слою
        hidden_inputs = numpy.dot(self.wih, inputs)
        # вычисляем сигнал после скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # вычисляем сигнал к выходному слою
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # вычисляем сигнал после выходного слоя
        final_outputs = self.activation_function(final_inputs)
        
        # вычисляем выходную ошибку 
        output_errors = targets - final_outputs
        # вычисляем ошибку на скрытом слое
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        # обновляем значения весов между скрытым и выходным слоями
        # Для сигмойды
        #self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        # Для гиперболического тангенса
        self.who += self.lr * numpy.dot((output_errors *(1.0 - pow(final_outputs,2))), numpy.transpose(hidden_outputs))
        # обновляем значения весов между входным и скрытым слоями
        # Для сигмойды
        #self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        # Для гиперболического тангенса
        self.wih += self.lr * numpy.dot((hidden_errors * (1.0 - pow(hidden_outputs,2))), numpy.transpose(inputs))
        pass
    
    # Метод для подсчёта совершенных эпох
    def count_ep(self): 
        self.count_epochs+=1
    
    # Метод опроса нейронной сети
    def query(self, inputs_list):
        # преобразуем входные значения из списка в 2-й массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # вычисляем сигнал к скрытому слою
        hidden_inputs = numpy.dot(self.wih, inputs)
        # вычисляем сигнал после скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # вычисляем сигнал к выходному слою
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # вычисляем сигнал после выходного слоя
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
    
    # Метод создания информации о последнем проходе обучения
    def information(self,inf_zero,inf_one,inf_two,inf_three):
        self.list_inf = [inf_zero,inf_one,inf_two,inf_three]
    
    # Метод получения информации о предыдущем проходе этой НС
    #(печать с помощью цикла)    
    def get_information(self):
        for inf in self.list_inf:
            print(inf)
    
    # Метод сохранения обученной модели
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

    # Метод резервного сохранения обученной модели
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
            
    # Метод загрузки обученной модели
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

flag_shuf = True # флаг перемешивания данных обучения
list_peremen = list(numpy.arange(0.1, 0.5, 0.05))
flag_boxplot = list_peremen[-1]
list_all_perf = []
for change in list_peremen:    
    # число входных, скрытых и выходных нейронов
    input_nodes = 15
    hidden_nodes = 20
    output_nodes = 2
    # Коэффициент обучения
    learning_rate =  float(round(change,2))
    #learning_rate = 0.15
    # Счётчик обучения
    count_lr = 0
    # Массив коэффициентов обучения по мере убывания
    #list_lr_down = numpy.arange(0.02,0.01,-0.005)
    # количество эпох обучения
    epochs = 20
    alfa = 0.77
    peremen_str = '+' + str(float(round(change,2)))
    
    # Имя созданной НС(если была создана)
    key = ''

    print('Start initialization!')
    # Если существует созданная НС,то она загружется из файла имя(key)
    if not key=='':
        n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate,alfa,key)
        print('Saving model is download!')
    # Если не существует созданная НС,то она создаётся заново
    else:
        n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate, alfa)
        print('Model is create!')
    # Подготовка данных
    if flag_shuf:
        training_data,test_data,flag_shuf = data_array.main(input_nodes,flag_shuf)
    #n.a = float(round(change,2))
    # Создание необходимых для работы пустых списков
    list_time = []
    list_perf = []
    list_epochs = []
    # Главный цикл соответствующий количеству эпох
    for e in range(epochs):
        # задаём коэффициент обучения
    #    try:
    #        n.lr = list_lr_down[count_lr]
    #        count_lr+=1
    #    except IndexError:
    #        count_lr = 0
    #        n.lr = list_lr_down[count_lr]
        # устанавливаем счётчик совпадений
        count_coincidence = 0
        # выводим строку при старте обучения
        print('Start epoch - ',n.count_epochs + 1)
        # запускаем таймер для вычисления времени, затраченного на обучение
        start = time.monotonic()
        # перемешиваем массив тренировочных данных
        numpy.random.shuffle(training_data)
        # цикл обучения
        for record in training_data:
            # формируем входной образ
            all_values = list(record)
            inputs = numpy.asfarray(all_values[1:])
            # создаём массив выходных занчений в соответствии с указанными маркерами
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            # обучаем сеть
            n.train(inputs, targets)
            pass
        # обновляем количество эпох для этой сети
        n.count_ep()
        pass
        # Создание необходимых для работы пустых списков
        scorecard = []
        list_label = []
        list_answers = []
        list_error = []

        print('Start check!')
        # перемешиваем массив тестовых данных
        numpy.random.shuffle(test_data)
        # цикл тестирования
        for record in test_data:
            # формируем входной образ
            all_values = list(record)
            inputs = numpy.asfarray(all_values[1:])
            # считываем маркер верного ответа из образа
            correct_label = int(all_values[0])
            # опрашиваем нейронную сеть
            outputs = n.query(inputs)
            label = numpy.argmax(outputs)
            # сравниваем ответ НС с маркером входного образа
            if (label == correct_label):
                # если ответ верный, то добавляем 1 к списку верных ответов
                scorecard.append(1)
                # если ответ верный, то добавляем 1 счётчику совпадений
                count_coincidence+=1
            else:
                # если ответ неверный, то добавляем 0 к списку верных ответов
                scorecard.append(0)
            pass
        # Вычисляем процент совпадений, т.е. точность
        scorecard_array = numpy.asarray(scorecard)
        perf = scorecard_array.sum() / scorecard_array.size
        perf = round(float(perf),5) * 100
        #Вывод на экран времени, затраченного на тренировку 
        result = time.monotonic() - start
        list_time.append(result)
        time_array = numpy.asarray(list_time)
        minutes = time_array.sum() // 60
        secounds = time_array.sum() % 60
        # вывод времени затраченного на обучение и тестирование
        print ("Time on the train and test = ", 
               format(minutes,'.0f'),'minute',
               format(secounds,'.3f'),'secounds')
        # выводим данные о количестве эпох, коэффициенте обучения и точности
        print('Epochs = ',n.count_epochs,'Learnning rate = ',n.lr)
        print('Coincidence = ',count_coincidence,'of',len(scorecard))
        print ("Performance = ", perf)
        #Создание строк для записи в файл НС
        line_zero = '\n'+ 'Time on the train and test = '+\
                     str(format(minutes,'.0f')) + ' minute ' + \
                     str(format(secounds,'.3f')) + ' secounds ' + '\n'
        line_one = 'Epochs = ' + str(n.count_epochs) + ' Learnning rate = ' + str(learning_rate) + '\n'
        line_two = 'Coincidence = ' + str(count_coincidence) +' of ' + str(len(scorecard))+ '\n'
        line_three = "Performance = " + str(perf)
        # Запись информации к объекту НС
        n.information(line_zero,line_one,line_two,line_three) 
        # Запись файла НС
        name = 'tan' + str(perf) + peremen_str + '+ep' + str(e)
        n.save_backup(name)
        list_perf.append(perf)
        list_epochs.append(e)
    list_all_perf.append(list_perf)
    list_error_perf = []
    flag = 0
    try:
        for x in range(len(list_perf)):
            temp = list_perf[x+1]-list_perf[x]
            list_error_perf.append(round(temp,3))
    except:
        #list_all_perf.append(list_error_perf)
        pass
    # Создаём и строим график зависимости
    #количества эпох от точности
    plt.plot(list_epochs,list_perf)
    str_nodes = 'input_nodes = ' + str(input_nodes) + ' ' +\
                'hidden_nodes = ' + str(hidden_nodes) +' ' +\
                'output_nodes = ' + str(output_nodes)
    plt.title(str_nodes)
    plt.xlabel('Количество эпох')# Подпись оси Х
    plt.ylabel('Точность, %')# Подпись оси Y
    plt.grid(True)# Включаем сетку на графике
    if len(peremen_str) != 5:
        peremen_str+='0'
    name_grapf = 'Train_trand ' + peremen_str +'.png'
    plt.savefig(name_grapf,dpi = 800)# Сохраняем график точности НС
    plt.show()
    shutil.move(name_grapf,BACKUP_PATH)
    plt.close()
    #plt.plot(numpy.arange(0,len(list_error_perf)),list_error_perf)
    
    # Строим BoxWhisker
    if flag_boxplot==change:
        plt.boxplot(list_all_perf,vert = False )
        plt.grid(True)
        plt.savefig('BoxWithWhisker.png',dpi = 800)
        plt.show()
        plt.close()
        # Создаем словарь со статистическими данными выборки
        dict_box = plt.boxplot(list_all_perf,vert = False )
        list_boxes = dict_box['boxes']
        list_medians = dict_box['medians']
        list_caps = dict_box['caps']
        list_fliers = dict_box['fliers']
        # Записываем статистические данные в файл txt
        file_stat = open('ReadMe.txt','a')
        for st in range(len(list_medians)):
            file_stat.write('График '+str(st+1) + '\n')
            file_stat.write('Границы: min - ')
            file_stat.write(str(round(list_boxes[st]._x[0],3)) + ' max - ')
            file_stat.write(str(round(list_boxes[st]._x[2],3)))
            file_stat.write('\n')
            file_stat.write('Медиана - ')
            file_stat.write(str(round(list_medians[st]._x[0],3)))
            file_stat.write('\n')
            file_stat.write('Выбросы - ')
            file_stat.write(str(list_fliers[st]._x))
            file_stat.write('\n')
            pass
        file_stat.write('------------------------\n')
        for st in range(len(list_caps)):
            if st==0:
                file_stat.write('График ' + str(st+1) + '\n')
                file_stat.write('min - ')
                file_stat.write(str(round(list_caps[st]._x[0],3)))
                file_stat.write('\n')
            elif st==1:
                file_stat.write('max - ')
                file_stat.write(str(round(list_caps[st]._x[0],3)))
                file_stat.write('\n')
            elif (st % 2)==0:
                file_stat.write('График ' + str(int(st/2+1)) + '\n')
                file_stat.write('min - ')
                file_stat.write(str(round(list_caps[st]._x[0],3)))
                file_stat.write('\n')
            elif (st % 2)!=0:
                file_stat.write('max - ')
                file_stat.write(str(round(list_caps[st]._x[0],3)))
                file_stat.write('\n')
            pass
        file_stat.close()
        shutil.move('ReadMe.txt',BACKUP_PATH)
        shutil.move('BoxWithWhisker.png',BACKUP_PATH)
# Перенос созданного графика в папку с архивом НС
#shutil.move('Train_trand.png',BACKUP_PATH)
