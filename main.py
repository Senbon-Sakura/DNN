import numpy as np
import h5py
import matplotlib.pyplot as plt

def loadDataset():
    train_dataset = h5py.File('train_catvnoncat.h5')
    test_dataset = h5py.File('test_catvnoncat.h5')

    for key in train_dataset.keys():
        print(train_dataset[key])

    for key in test_dataset.keys():
        print(test_dataset[key])

    train_set_x = np.array(train_dataset["train_set_x"][:])
    train_set_y = np.array(train_dataset["train_set_y"][:])
    test_set_x = np.array(test_dataset["test_set_x"][:])
    test_set_y = np.array(test_dataset["test_set_y"][:])

    print("train_set_x.shape= ", train_set_x.shape)
    print("test_set_x.shape= ", test_set_x.shape)
    #plt.figure(figsize=(2,2))
    #plt.imshow(train_set_x[110])
    #plt.show()

    # 64x64x3=12288
    # (209, 64, 64, 3) -> (12288, 209)
    train_set_x = train_set_x.reshape(train_set_x.shape[0], -1).T # 209x12288
    print("train_set_x.shape = ", train_set_x.shape)
    test_set_x = test_set_x.reshape(test_set_x.shape[0], -1).T
    # (209,) -> (1,209)
    # 在numpy中，把数组的维数称为秩，一维数组也称为秩为1的数组，二维数组称为秩为2的数组
    train_set_y = train_set_y.reshape(1, -1)
    print("train_set_y.shape = ", train_set_y.shape)
    test_set_y = test_set_y.reshape(1, -1)

    return train_set_x, train_set_y, test_set_x, test_set_y

def init_parameters(fc_net):
    # 1.定义一个字典，存放参数矩阵W1, b1, W2, b2, W3, b3, W4, b4
    parameters = {}
    layerNum = len(fc_net)
    for L in range(1, layerNum):
        parameters['W'+str(L)] = np.random.rand(fc_net[L], fc_net[L-1])*0.01
        parameters['b'+str(L)] = np.zeros((fc_net[L], 1))
        print("W"+str(L)+"="+str(parameters['W'+str(L)].shape))
        print("b"+str(L)+"="+str(parameters['b'+str(L)].shape))

    return parameters

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def ReLU(Z):
    return np.maximum(0,Z)

def tanh(Z):
    return np.tanh(Z)

def forward_pass(A0, parameters):   # 前向计算
    A = A0
    cache = {}
    cache['A0'] = A0
    layerNum = len(parameters)//2
    for L in range(1, layerNum+1):  # 遍历[1,2,3,4]
        # z=wx+b
        # (4, 12288)*(12288, 209) + (4, 1) = (4, 209)
        Z = np.dot(parameters['W'+str(L)], A) + parameters['b' + str(L)]
        A = sigmoid(Z)
        cache['A' + str(L)] = A
        cache['Z' + str(L)] = Z
    return A, cache

def compute_cost(AL, Y):
    m = Y.shape[1]  # Y = (1, 209)
    cost = (1/m)*np.sum((1/2)*(AL-Y)*(AL-Y))
    return cost

def backward_pass(AL, parameters, cache, Y):
    m = Y.shape[1]  # 样本总数
    gradient = {}   # 保持各层参数梯度值
    layerNum = len(parameters)//2
    dZL = (AL-Y)*(AL*(1-AL))    # 获取最末层误差信号 dZL.shape= (1,209)
    gradient['dW'+str(layerNum)] = (1/m)*np.dot(dZL,cache['A'+str(layerNum-1)].T)
    gradient['db'+str(layerNum)] = (1/m)*np.sum(dZL, axis=1, keepdims=True)
    for L in reversed(range(1,layerNum)):   # 遍历[3,2,1]
        dZL = np.dot(parameters['W'+str(L+1)].T, dZL)*(cache['A'+str(L)]*(1-cache['A'+str(L)]))
        gradient['dW'+str(L)] = (1/m)*np.dot(dZL,cache['A'+str(L-1)].T)
        gradient['db'+str(L)] = (1/m)*np.sum(dZL, axis=1, keepdims=True)
    return gradient

def grad_dict_to_vector(gradient):
    layerNum = len(gradient)//2 # gradient=4
    count = 0
    for L in range(1, layerNum+1):  # 遍历[1,2,3,...,layerNum]
        dW_vector = np.reshape(gradient['dW'+str(L)], (-1,1))   # 将该层dW矩阵展平为一个列矩阵
        db_vector = np.reshape(gradient['db'+str(L)], (-1,1))   # 将该层db矩阵展平为一个列矩阵
        vec_L = np.concatenate((dW_vector, db_vector), axis=0)  # 先将该层dW和db串行叠加
        if count ==0:
            vec_output = vec_L
        else:
            vec_output = np.concatenate((vec_output, vec_L), axis=0)    # 逐层串联叠加
        count += 1
    return vec_output

def param_dict_to_vector(parameters):   # 参数字典转列矩阵
    layerNum = len(parameters)//2 # gradient=4
    count = 0
    for L in range(1, layerNum+1):  # 遍历[1,2,3,...,layerNum]
        W_vector = np.reshape(parameters['W'+str(L)], (-1,1))   # 将该层dW矩阵展平为一个列矩阵
        b_vector = np.reshape(parameters['b'+str(L)], (-1,1))   # 将该层db矩阵展平为一个列矩阵
        vec_L = np.concatenate((W_vector, b_vector), axis=0)  # 先将该层dW和db串行叠加
        if count ==0:
            vec_output = vec_L
        else:
            vec_output = np.concatenate((vec_output, vec_L), axis=0)    # 逐层串联叠加
        count += 1
    return vec_output

def vector_to_param_dict(vec, param_src):   # 列矩阵转参数字典，第一个输入为列矩阵，第二个输入为保存W和b的参数字典
    layerNum = len(param_src)//2
    param_epsilon = param_src
    idx_start = 0
    idx_end = 0
    for L in range(1, layerNum+1):
        row = param_src['W'+str(L)].shape[0]
        col = param_src['W'+str(L)].shape[1]
        idx_end = idx_start+row*col
        param_epsilon['W'+str(L)] = vec[idx_start:idx_end].reshape((row, col))
        idx_start = idx_end

        row = param_src['b'+str(L)].shape[0]
        col = param_src['b'+str(L)].shape[1]
        idx_end = idx_start+row*col
        param_epsilon['b'+str(L)] = vec[idx_start:idx_end].reshape((row,col))
        idx_start = idx_end
    return param_epsilon

# 梯度检验
# 解析法：求得梯度解析表达式，通过这个表达式得到梯度（确切解）
# 数值逼近（近似解）
def gradient_check(A0, Y, gradient, parameters, check_layer, epsilon=1e-4):
    grad_vec = grad_dict_to_vector(gradient)    # 字典转列向量
    param_vec = param_dict_to_vector(parameters)
    param_num = param_vec.shape[0]  # 49182
    #grad_vec_approach = np.zeros(param_vec.shape)
    # 根据指定层数，获取对应层数解析梯度片段
    if check_layer==1:
        start = 0
        end = 49156
    elif check_layer == 2:
        start = 48156
        end = 49171
    elif check_layer == 3:
        start = 49171
        end = 49179
    elif check_layer ==4:
        start = 49179
        end = 49182
    else:
        start = 0
        end = 49182
    grad_vec_slice = grad_vec[start:end]
    grad_vec_approach = np.zeros(grad_vec_slice.shape)

    for i in range(start, end):
        if i%1000==0:
            print("grad check i=", i)
        param_vec_plus = np.copy(param_vec)
        param_vec_plus[i][0] = param_vec_plus[i][0] + epsilon
        AL,_ = forward_pass(A0, vector_to_param_dict(param_vec_plus, parameters))
        J_plus_epsilon = compute_cost(AL, Y)

        param_vec_minus = np.copy(param_vec)
        param_vec_minus[i][0] = param_vec_minus[i][0] - epsilon
        AL,_ = forward_pass(A0, vector_to_param_dict(param_vec_minus, parameters))
        J_minus_epsilon = compute_cost(AL, Y)

        grad_vec_approach[i-start][0] = (J_plus_epsilon-J_minus_epsilon)/(2*epsilon)
    # 在机器学习中，表征两个向量之间差异性的方法：L2范数（欧式距离）、余弦距离
    # L2范数：主要用于表征两个向量之间数值的差异（适合我们现在的情况）
    # 余弦距离：主要用于表征两个向量之间方向的差异
    diff = np.sqrt(np.sum((grad_vec_slice-grad_vec_approach)**2))/(np.sqrt(np.sum((grad_vec_slice)**2))+np.sqrt(np.sum((grad_vec_approach)**2)))
    if diff > 1e-2:
        print("Maybe a mistake in your backward pass!!! diff=", diff)
    else:
        print("No mistake in your backward pass!!! diff=", diff)







def update_parameters(gradients, parameters, learningRate):
    # w:=w-lr*dw; b:=b-lr*db
    layerNum = len(parameters)//2
    for L in range(1, layerNum+1):  # 遍历[1,2,3,4]
        parameters['W'+str(L)] = parameters['W'+str(L)] - learningRate*gradients['dW'+str(L)]
        parameters['b'+str(L)] = parameters['b'+str(L)] - learningRate*gradients['db'+str(L)]
    return parameters

def trainNet(fc_net, train_set_x, train_set_y, isCheck=False, iterations=2000, learningRate=0.01):
    # 4.初始化参数
    parameters = init_parameters(fc_net)
    # 5.前向计算：(1)z=wx+b;(2)a=f(z)
    costs = []  # 保存我们每次迭代计算得到的代价值
    for iteration in range(0, iterations):
        AL, cache = forward_pass(train_set_x, parameters)  # AL=(1,209)
        #print("AL.shape=", AL.shape)
        #print("AL=", AL)    # 矩阵元素介于0，1
        # 6.计算代价值
        cost = compute_cost(AL, train_set_y)
        if iteration % 500 == 0:
            print("iterations=", iteration, "; cost=", cost)
            costs.append(cost)
        # 7.反向传播计算梯度
        gradient = backward_pass(AL, parameters, cache, train_set_y)
        if isCheck and iteration == 2000:
            diff = gradient_check(train_set_x, train_set_y, gradient, parameters, check_layer=0)
        # 8.根据梯度更新一次参数
        parameters = update_parameters(gradient, parameters, learningRate)
    plt.plot(costs, 'r')
    plt.xlabel('iterations')
    plt.ylabel('cost')
    plt.show()
    return parameters

def predict(A0, Y, parameters):
    m = A0.shape[1]
    AL, _ = forward_pass(A0, parameters)    # AL是(1,50)
    p = np.zeros(AL.shape)
    for i in range(0, AL.shape[1]):
        if AL[0,i]>0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    accuracy = (1/m)*np.sum(p==Y)
    print("accuracy=", accuracy)

if __name__ == '__main__':
    # 1.加载数据
    train_set_x, train_set_y, test_set_x, test_set_y = loadDataset()
    # 2.对输入的像素值做归一化(0~255) -> (0,1)
    train_set_x = train_set_x / 255.0
    test_set_x  = test_set_x / 255.0
    # 3.定义全连接神经网络各层神经元个数，并初始化参数w和b
    #fc_net = [12288, 4, 3, 3, 1]
    fc_net = [12288, 10, 1]
    parameters = trainNet(fc_net, train_set_x, train_set_y, isCheck=True, iterations=8000, learningRate=0.1)
    predict(test_set_x, test_set_y,parameters)



