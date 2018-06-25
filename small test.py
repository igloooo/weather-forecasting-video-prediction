import numpy as np
import matplotlib.pyplot as plt
import torch
from data_generator import *
import time
import train
'''
arr = np.load("mnist_test_seq.npy")
print(len(arr[0,:,0,0]))
batch0 = arr[:,0,:,:]
print(len(batch0[0]))
print(len(batch0[0][0]))
plt.figure(2, figsize=(20, 1))
plt.clf()
for i in range(20):
    plt.subplot(1, 20, i + 1)
    plt.imshow(batch0[i, :, :], cmap=plt.cm.gray, interpolation="nearest")
    plt.axis('off')
    plt.draw()
plt.show()

loss1 = torch.nn.MSELoss()
loss2 = torch.nn.L1Loss()
criterion = lambda inputs, targets: loss1(inputs, targets) + loss2(inputs, targets)
inputs =torch.tensor(batch0[0:10], dtype=torch.float32, requires_grad=True)
targets = torch.tensor(batch0[10:20], dtype=torch.float32, requires_grad=False)
loss = criterion(inputs, targets)
print(loss.size())
print(loss.item())
loss.backward()
print(inputs.grad.size())
print(inputs.grad)

a = torch.randn(1, 1, 3, 3, requires_grad=True)
upsample = torch.nn.Upsample((6,6))
b = upsample(a)
c = b*torch.randn(1, 1, 6, 6)
c.sum().backward()
print(a.grad)
print(b.grad_fn)
'''
'''
for i in range(10):
    im = plt.imread('mnist/raw digit/'+str(i)+'.png')
    im = im[:,:,1]
    im = 1-im
    im.reshape(1,28,28)
    np.save('mnist/raw digit/'+str(i)+'.npy', im)
'''
'''
generator = BouncingMNISTDataHandler(8, 2)
data = generator.GetBatch()
plt.figure(figsize=(20,8))
for i in range(8):
    for j in range(20):
        plt.subplot(8, 20, 20*i+j+1)
        plt.imshow(data[j,i,0,:,:])
plt.show()
'''
