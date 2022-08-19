# Программа для создания выборки данных
# Библиотека для массивов
import numpy as np

def main(nodes,flag):
    #nodes = 15
    # Общее время переходного процесса
    const_t = 4
    # Находим шаг времени в зависимости от числа вх.нейронов
    numbers_t = const_t / nodes
    # Создаём временный список
    temp_list = []
    # Создаём лист значений X
    list_t = np.arange(0.1,const_t,numbers_t)
    
    # Апериодическое звено
    # Создаём лист значений k (коэффицента усиления)
    list_k = list(np.arange(0.1,0.98,0.02))
    # Создаём лист Т (постоянная времени)
    lconst_t = list(np.arange(0.1,0.98,0.01))# было 0.02
    # Цикл для создания маркированных данных
    for x in lconst_t:
        for k in list_k:
            # Находим значения Y
            list_y = k *(1-np.exp((-list_t)/x))
            # Добавляем маркер 0
            list_y = np.insert(list_y,0,0)
            temp_list.append(list_y)      
    
#    # Интегрирующее звено
#    # Создаём лист значений k (коэффицента наклона для прямой)
#    list_k = list(np.arange(0,1,0.0003))# было 0.001
#    # Создаём пустой лист значений Y
#    list_y =[]
#    # Цикл для создания маркированных данных
#    for k in list_k:
#        for x in list_t:
#            # Находим значения Y
#            if x <= 0.9:
#                y = k * x
#            else:
#                y = list_y[-1]
#            list_y.append(y)
#        # Добавляем маркер 0
#        list_y.insert(0,0)
#        list_y_arr = np.array(list_y)
#        temp_list.append(list_y_arr)
#        list_y =[]
    
    # Колебательное звено
    # Создаём лист значений T постоянной времени и k (коэффицента усиления)
    list_T_k = list(np.arange(0.1,0.8,0.01))
    # Создаём список коэффициента затухания
    list_const_epsila = list(np.arange(0.3,0.9,0.01))
    # Создаём лист значений Y
    list_y =[]
    # Цикл для создания маркированных данных
    for i in range(len(list_T_k)):
        k = list_T_k[i]
        T = list_T_k[i]
        for const_epsila in list_const_epsila:
            for x in list_t:
                # Определяем константу в уравнении звена
                const_alfa = pow(1-const_epsila,-0.5)
                # Находим значения Y
                y = k*(1-np.exp(-(const_epsila*x)/T)/const_alfa*np.sin(const_alfa*x/T+np.arctan(const_alfa/const_epsila)))
                list_y.append(y)
            # Добавляем маркер 1
            list_y.insert(0,1)
            list_y_arr = np.array(list_y)
            temp_list.append(list_y_arr)
            list_y =[]
    # Определяем выходной массив
    data_arr = np.array(temp_list)
    # Перемешиваем выходной массив
    np.random.shuffle(data_arr)
    # Делим выходной массив на 2 равные части
    try:
        data_arr = np.split(data_arr,2,axis=0)
    except ValueError:
        data_arr = np.delete(data_arr,-1,axis=0)        
        data_arr = np.split(data_arr,2,axis=0)
    data1 = data_arr[0]
    data2 = data_arr[1]
    flag = False
    # Возвращаем массив в главную программу
    return data1,data2,flag
# Команды для отладки
d,s,f = main(15,True)
