import numpy as np
import pickle
def generate_time_series(n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))
    series += 0.1 * (np.random.rand(n_steps) - 0.5)
    return series

with open("7000-test-set.pkl", "wb") as f:
    testSet = []
    for i in range(7000):
        data = generate_time_series(51)
        
        inp = np.zeros((50, 1))
        for i in range(0, len(data)-1):
            inp[i, 0] = data[i]
        out = np.array([[data[50:51][0]]])
        inpOut = (inp, out)
        testSet.append(inpOut)
    pickle.dump(testSet, f)

with open("2000-test-set.pkl", "wb") as f:
    testSet = []
    for i in range(2000):
        data = generate_time_series(51)
        
        inp = np.zeros((50, 1))
        for i in range(0, len(data)-1):
            inp[i, 0] = data[i]
        out = np.array([[data[50:51][0]]])
        inpOut = (inp, out)
        testSet.append(inpOut)
    pickle.dump(testSet, f)


def niave_guessing():
    error = 0
    with open("2000-test-set.pkl", "rb") as f:
        tSet = pickle.load(f)
        print(tSet[0])
        for i in tSet:
           error += (i[1][0]-i[0][-1])**2
    return error/2000
print(niave_guessing())