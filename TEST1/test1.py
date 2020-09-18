import numpy as np
import torch
import random
import math

def vectorize_sumproducts(a, b):
    return np.sum(a * b)

def vectorize_Relu(a):
    a[a <= 0] = 0
    return a

def vectorize_PrimeRelu(a):
    a[a <= 0] = 0
    a[a > 0] = 1
    return a

######################################
#3 Variable length
######################################

def Slice_fixed_point(a, begin, ind):
    try:
        List = []
        for i in a:
            List2 = []
            for j in i:
                List2.append(j[begin - 1: begin + ind])
            List.append(List2)
        return np.array(List) 
    except: return -1

def slice_last_point(a, end):
    try:
        List = []
        for i in a:
            List2 = []
            for j in i:
                List2.append(j[-end: ])
            List.append(List2)
        return np.array(List)
    except: return -1

def pad_pattern_end(a):
    for i in a:
        maxNum = 0
        for j in i:
            if maxNum < len(j): maxNum = len(j)
        for j in i:
            while(maxNum > len(j)):
                length = len(j)
                for k in range(maxNum - length):
                    if length - 1 - k < 0: break
                    j.append(j[length - 1 - k])
    return a
def pad_constant_central(a):
    for i in a:
        maxNum = 0
        for j in i:
            if maxNum < len(j): maxNum = len(j)
        for j in i:
            while(1):
                if len(j) == maxNum: break
                j.append("cval")
                if len(j) == maxNum: break
                j.insert(0, "cval")
    return a

#######################################
#PyTorch
#######################################

# numpy&torch

def numpy2tensor(a):
    return torch.from_numpy(a)
def tensor2numpy(a):
    return a.numpy()

#Tensor Sum-products

def Tensor_Sumproducts(a, b):
    return torch.sum(a * b)

#Tensor ReLu and ReLu prime

def Tensor_Relu(a):
    return torch.clamp(numpy2tensor(a), min = 0)
def Tensor_Relu_prime(a):
    a = Tensor_Relu(a)
    one = torch.ones_like(a)
    return torch.where(a > 0, one, a)

if __name__ == "__main__":
    a = np.array([[1,  2, 3,  4]])
    b = np.array([[1,  2, 3,  4]])
    print("vectorize_sumproducts: ", vectorize_sumproducts(a, b))
    c = np.random.randn(5, 5)
    print("vectorize_Relu:\n", vectorize_Relu(c))
    c = np.random.randn(5, 5)
    print("vectorize_PrimeRelu:\n", vectorize_PrimeRelu(c))
    d = [[[1, 2, 3, 4, 5, 7], [2, 3, 4, 5], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6]], [[1, 2, 3, 4, 5, 7], [2, 3, 4, 5], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6]]]
    print("Slice_fixed_point:\n", Slice_fixed_point(d, 2, 1))
    d = [[[1, 2, 3, 4, 5, 7], [2, 3, 4, 5], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6]], [[1, 2, 3, 4, 5, 7], [2, 3, 4, 5], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6]]]
    print("slice_last_point:\n", slice_last_point(d, 3))
    d = [[[1, 2], [2, 3, 4, 5], [2, 3, 4, 5, 6], [2, 3, 4, 6]], [[1, 2, 3, 4, 5, 7], [4, 5], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6]]]
    print(pad_pattern_end(d))
    d = [[[1, 2, 3, 4], [2, 3, 4, 5], [2, 3, 4, 5, 6], [2, 6]], [[1, 2, 3, 4, 5, 7], [2, 4, 5], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6]]]
    print(pad_constant_central(d))
    e = np.ones(5)
    print("numpy2tensor:\n", numpy2tensor(e))
    e = torch.ones(5)
    print("tensor2numpy:\n", tensor2numpy(e))
    f1 = torch.tensor([[1,  2, 3,  4]])
    f2 = torch.tensor([[1,  2, 3,  4]])
    print(Tensor_Sumproducts(f1, f2))
    g = np.random.randn(5, 5)
    print("Tensor_Relu:\n", Tensor_Relu(g))
    g = np.random.randn(5, 5)
    print("Tensor_Relu_prime:\n", Tensor_Relu_prime(g))