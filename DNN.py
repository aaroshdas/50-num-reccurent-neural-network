import numpy as np
import pickle

def convert_to_tuple(num, num_of_digits):
    t1 = bin(num)
    t1 = t1[2:]
    while len(t1) < num_of_digits:
        t1 = "0" + t1
    t2 = list(t1)
    t3 = tuple(int(x) for x in t2)
    return t3
def create_truth_table(num_of_bits, output):
    out = []
    binary_num = convert_to_tuple(output, 2**num_of_bits)
    for i in range(2**num_of_bits):
        out.append((convert_to_tuple(i, num_of_bits), binary_num[2**num_of_bits-1-i]))
    return out[::-1]

#in check, num of bets is length of w (weight)

def step(num):
    if(num >0):
        return 1
    return 0



step_vectorized = np.vectorize(step)

def perceptron(A, w, b, x):
    sum1 = 0
    for i in range(len(w)):
        sum1 += w[i] * x[i]
    return A(sum1+b)
def check(n, w, b):
    correct= 0
    table = create_truth_table(len(w), n)
    for i in table:
        if(perceptron(step, w, b, i[0]) == i[1]):
            correct +=1
    return correct/2**len(w)
def train(num_of_bits, target):
    table = create_truth_table(num_of_bits, target)
    lastEpoch=((0,)*num_of_bits, 0)
    currentEpoch = lastEpoch
    for epoch in range(100):
        for row in table:
            error = row[1]-perceptron(step, currentEpoch[0], currentEpoch[1], row[0])
            currentEpoch = (tuple(currentEpoch[0][i] + (error * row[0][i]) for i in range(len(currentEpoch[0]))), currentEpoch[1]+error)
        if(currentEpoch == lastEpoch):
            break
    return check(target, currentEpoch[0], currentEpoch[1]), currentEpoch

def check_every_number(num_bits):
    correct =0
    for k in range(2**(2**num_bits)):
        if(train(num_bits, k) == 1):
            correct +=1
    print(correct, (2**(2**num_bits)))

def xor(in1):
    p3 = perceptron(step, (1,1), 0, in1)
    p4 = perceptron(step, (-1,-2), 3, in1)
    p5 = perceptron(step, (1,2), -2, (p3, p4))
    return p5

def p_net(a_vec, weights,biases, input):
    outputs = []
    outputs.append(input)
    for layer in range(1, len(weights)):   
        outputs.append(a_vec(weights[layer] @ outputs[layer-1] + biases[layer]))
    return outputs[-1]


def check_circle(input):
    if((input[0][0]**2+ input[1][0]**2)**0.5 < 1):
        return 1
    return 0


def activationFuncDerivative(x):
    return np.cosh(x)

def test_func(inputs, w, b, activationFunc):
    correctRatio = 0
    for inp in inputs: #inputs (0,1 stuff like that)
        As ={}
        As[0] = inp[0]
        dots = {}        
        for layer in range(1, len(w)): #get layer
            dots[layer] = (w[layer]@As[layer-1])+b[layer]
            As[layer] = activationFunc(dots[layer])
    
        correctRatio +=(inp[1][0,0]-As[len(w)-1][0,0])**2
    return correctRatio/len(inputs)
       
def back_propagation(inputs, w, b, activationFunc, learningRate, epochs):
    for epoch in range(epochs):     
        for ind, inp in enumerate(inputs): #inputs (0,1 stuff like that)
            if(ind%2000 == 0):
                print(ind)
            As ={}
            As[0] = inp[0]
            dots = {}        
            for layer in range(1, len(w)): #get layer
                dots[layer] = (w[layer]@As[layer-1])+b[layer]
                As[layer] = activationFunc(dots[layer])
            deltas= {}
            deltas[len(w)-1] = activationFuncDerivative(activationFunc, dots[len(w)-1]) * (inp[1]-As[len(w)-1])
            
            for layer in range(len(w)-2, 0,-1):
                deltas[layer] = activationFuncDerivative(activationFunc, dots[layer]) *(np.transpose(w[layer+1])@deltas[layer+1])
            for layer in range(1, len(w)):
                b[layer] = b[layer]+learningRate*deltas[layer]
                w[layer] = w[layer]+learningRate*deltas[layer] *np.transpose(As[layer-1])
            
    return (w,b)

inputs = []
with open("7000-test-set.pkl", 'rb') as f:
        data = pickle.load(f)
        for i in data:
            inputs.append(i)
tests = []
with open("2000-test-set.pkl", 'rb') as f:
        data = pickle.load(f)
        for i in data:
            tests.append(i)
def create_rand_values(dimensions):
    weights= [None]
    biases = [None]
    for i in range(1,len(dimensions)):
        weights.append(2*np.random.rand(dimensions[i],dimensions[i-1]) - 1)
        biases.append(2*np.random.rand(dimensions[i],1)-1)
    return weights, biases


w1,b1 = create_rand_values([50,1])

# with open("weights_and_biases.pkl", "wb") as f:
#     pickle.dump((w,b), f)

# with open("weights_and_biases.pkl", "rb") as f:
#     w1,b1 = pickle.load(f)

print(test_func(tests, w1, b1, np.tanh))
print("")
for i in range(100):
    w1, b1 = back_propagation(inputs, w1, b1, np.tanh, 0.1, 1)
    with open("weights_and_biases.pkl", "wb") as f:
        pickle.dump((w1,b1), f)
    print(test_func(tests, w1, b1, np.tanh))
    print("run number: " + str(i))
    print("")
print("Done running")

