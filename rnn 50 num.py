import numpy as np
import pickle




def activationFuncDerivative(x):
    return 1 / np.cosh(x) ** 2

def test_func(inputs, wL,wS, b, activationFunc):
    correctRatio = 0
    for inp in inputs: #inputs (0,1 stuff like that)
        As ={} 
        dots = {}
        for layer in range(1, len(wL)):
            As[layer, 0] = np.zeros((len(wL[layer]), 1))
        for step in range(1, len(inp[0])+1): 
            As[(0, step)] = np.array([inp[0][step-1]])
            for layer in range(1, len(wL)):
                dots[(layer, step)] =wL[layer]@As[(layer-1, step)] + wS[layer] @As[layer, step-1] +b[layer]
                As[(layer, step)] = activationFunc(dots[(layer, step)])
        
        correctRatio +=(inp[1]-As[(len(wL)-1, len(inp[0]))][0,0])**2
    return correctRatio/len(inputs)
       
def back_propagation(inputs, wL,wS, b, activationFunc, learningRate, epochs):
    for epoch in range(epochs):     
        for ind, inp in enumerate(inputs): #inp[input, output]
            if(ind%2000 == 0):
                print(ind)

            As ={} 
            dots = {}
            for layer in range(1, len(wL)):
                As[layer, 0] = np.zeros((len(wL[layer]), 1))

            for step in range(1, len(inp[0])+1): 
                As[(0, step)] = np.array([inp[0][step-1]])
      
                for layer in range(1, len(wL)):
                    # print("BELOW:")
                    # print(wS[layer])
                    # print(As[(layer, step-1)])
                    dots[(layer, step)] =wL[layer]@As[(layer-1, step)] + wS[layer] @As[(layer, step-1)] +b[layer]
                    As[(layer, step)] = activationFunc(dots[(layer, step)])

            deltas= {}
            deltas[(len(wL)-1, len(inp[0]))] = activationFuncDerivative(dots[(len(wL)-1, len(inp[0]))]) * (inp[1]-As[(len(wL)-1, len(inp[0]))])
            for step in range(len(inp[0])-1, 0,-1):
                deltas[(len(wL)-1, step)] = activationFuncDerivative(dots[len(wL)-1, step])*(np.transpose(wS[len(wL)-1])@deltas[len(wL)-1, step+1])
            for layer in range(len(wL)-2,0,-1):
                deltas[(layer, len(inp[0]))] = activationFuncDerivative(dots[(layer, len(inp[0]))])*(np.transpose(wL[layer+1])@deltas[(layer+1, len(inp[0]))])
            
            for step in range(len(inp[0])-1, 0,-1):
                for layer in range(len(wL)-2,0,-1):
                    deltas[(layer, step)]=activationFuncDerivative(dots[(layer, step)])*(np.transpose(wS[layer])@deltas[(layer, step+1)] )+ activationFuncDerivative(dots[(layer, step)])*(np.transpose(wL[layer+1])@deltas[(layer+1, step)])

            for layer in range(1, len(wL)):
                biasSum = 0
                weightLSum = 0
                weightSSum = 0
                for step in range(1, len(inp[0])+1):
                    biasSum += deltas[(layer, step)] # DELTAS ARE SO BIG ITS CAUSING OVERFLOW ERRORS THIS IS THE ERROR ******
                    weightLSum += deltas[(layer, step)] @ np.transpose(As[(layer-1, step)])
                    if(step!=1): #except for first
                        weightSSum += deltas[(layer, step)] @ np.transpose(As[layer, step-1])
     
                wL[layer] = wL[layer]+(learningRate*weightLSum) 
                wS[layer] = wS[layer]+(learningRate*weightSSum)
                b[layer]=b[layer]+(learningRate*biasSum)     
    print(test_func(tests, w1L,w1s, b1, np.tanh))
    return wS,wL, b

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
    weightStep= [None]
    biases = [None]
    weightsLayer=[None]
    for i in range(1,len(dimensions)):
        weightStep.append(2*np.random.rand(dimensions[i],dimensions[i]) - 1)
        weightsLayer.append(2*np.random.rand(dimensions[i],dimensions[i-1]) - 1)
        biases.append(2*np.random.rand(dimensions[i],1)-1)
    return weightStep,weightsLayer, biases


w1s,w1L, b1 = create_rand_values([1,6,1])
print(test_func(tests,w1L,w1s, b1, np.tanh))


w1s,w1L, b1= back_propagation(inputs, w1L, w1s, b1, np.tanh, 0.01, 2)
print("2 back prop done")


w1s,w1L, b1= back_propagation(inputs, w1L, w1s, b1, np.tanh, 0.01, 5)
print("5 back prop done")
