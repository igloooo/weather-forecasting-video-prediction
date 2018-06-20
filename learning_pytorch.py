import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import reduce
import random
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import cv2
from data_generator import BouncingMNISTDataHandler
from graphviz import Digraph

import logging
logging.basicConfig(level=logging.DEBUG)

'''
#initialize
x = torch.empty(5, 3)
x = torch.zeros(5, 3)
x = torch.ones(5, 3)
x = torch.tensor([[3, 3], [5, 3]])
x = torch.randn(3, 3)

#reinitialize
x = torch.randn_like(x, dtype=torch.double)
x = torch.zeros_like(x)

#tensor additions
x = torch.randn(3, 3, dtype=torch.double)
y = torch.randn(3, 3, dtype=torch.double)
z = x + y
z = torch.add(x, y)
z = torch.empty(3, 3, dtype=torch.double)
torch.add(x, y, out=z)
x.add_(y)

#matrix multiplication
# vector x vector
tensor1 = torch.randn(3)
tensor2 = torch.randn(3)
torch.matmul(tensor1, tensor2).size()  # torch.Size([])
# matrix x vector
tensor1 = torch.randn(3, 4)
tensor2 = torch.randn(4)
torch.matmul(tensor1, tensor2).size()  # torch.Size([3])
# batched matrix x broadcasted vector
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(4)
torch.matmul(tensor1, tensor2).size()  # torch.Size([10, 3])
#  batched matrix x batched matrix
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(10, 4, 5)
torch.matmul(tensor1, tensor2).size()  # torch.Size([10, 3, 5])
# batched matrix x broadcasted matrix
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(4, 5)
torch.matmul(tensor1, tensor2).size()  # torch.Size([10, 3, 5])

#accession, reshape
x[:, 1] = y[:, 1]
x = x.view(9)

#converting between numpy array and tensor
x = x.numpy()
x = torch.from_numpy(x)

#autograd
a = torch.randn(3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
c = torch.ones(3)
d = torch.matmul(a + b + 1, c)
d.backward(retain_graph=True)
# print(a.grad, b.grad)
b.detach_()
d.backward()
# print(a.grad, b.grad)
print(a.grad)
a.grad = torch.tensor([0., 0., 0.])
a.grad.requires_grad_()
e = torch.matmul(c, a.grad)
e.backward()
print(a.grad)

a = torch.ones(3, 3, 6, 6)
b = torch.randn(1, 3, 10, 10)
c = torch.nn.functional.conv2d(input=b, weight=a, padding=3)
print(c.size())
'''
'''
generator = BouncingMNISTDataHandler(1, 2)
seq, unused = generator.GetBatch()
seq = torch.tensor(seq, dtype=torch.float32)
seq = seq*255
result = cv2.calcOpticalFlowFarneback(seq[0, 0, 0].numpy(), seq[1, 0, 0].numpy(), None, 0.5, 13, 15, 3, 5, 1.2, 0)
plt.figure()
plt.subplot(2,2,1)
plt.imshow(result[:,:,0])
plt.subplot(2,2,2)
plt.imshow(result[:,:,1])
plt.subplot(2,2,3)
plt.imshow(seq[0,0,0].numpy())
plt.subplot(2,2,4)
plt.imshow(seq[1,0,0].numpy())
plt.show()
'''
'''
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, padding=2)
        self.conv2 = nn.Conv2d(20, 20, 5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))

model = Model()
for name, param in model.named_parameters():
    # print(name, type(param.data), param.size())
    nn.init.constant_(param, 0)
'''

