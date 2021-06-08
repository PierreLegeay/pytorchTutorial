import torch as t
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

def training(net,trainloader,device,optimizer,criterion):
    for epoch in range(3):  # loop over the dataset multiple times

        current_loss = 0.0
        for i, (inputs,labels) in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net.forward(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            current_loss += loss.item()
            if i % 1000 == 999:    # print every 1000 batch
                print(f'[Epoch : {epoch + 1}, Iteration : {i + 1}] Progress: {100*i*batch_size/len(trainloader.dataset)}% loss: {current_loss / 1000}')
                current_loss = 0.0
            if i % 10000 == 0: #saving model every 10k batch
                t.save(net.state_dict(), "./MNISTmodelEpoch"+str(epoch+1)+"iter"+str(i))

    print('Finished Training')

def testing(net,model_path,device,testloader,criterion):
    net.load_state_dict(t.load(model_path)) #loading model
    net.eval()
    test_loss = 0
    correct = 0
    with t.no_grad():
        for data, label in testloader:
            data, label = data.to(device), label.to(device)
            output = net.forward(data)

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            correct += pred.eq(label.view_as(pred)).sum().item() #if pred is correct incrementing
            test_loss += criterion(output, label)

    test_loss /= len(testloader.dataset) #Loss per mini-batch

    print('\nTest set: AvgLoss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format( #Calculating accuracy of the model
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  #1 inputs 32 ouputs Kernel = 5 
        self.conv2 = nn.Conv2d(32, 64, 3) 
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1600, 128) 
        self.fc2 = nn.Linear(128, 10)    

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) #pool after convolution
        x = self.pool(F.relu(self.conv2(x)))
        x = t.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x)) #Linear Transformation
        x = self.fc2(x) #10 classes output
        return F.log_softmax(x, dim=1)


path = "./dataMNIST"

#Setting cpu or gpu
device = t.device('cpu')
if t.cuda.is_available():
    device = t.device('cuda:6')

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

batch_size = 4

#Training set
trainset = tv.datasets.MNIST(root = path, train = True, download = True, transform = transform)
trainloader = t.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = 1)

#Testing set
testset = tv.datasets.MNIST(root= path, train=False, download = True, transform = transform)
testloader = t.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = True, num_workers = 1)

net = Net().to(device) #net instance

criterion = nn.CrossEntropyLoss() #Setting lose fuction
optimizer = optim.Adam(net.parameters(), lr=0.001) #Setting optimizer

#training(net,trainloader,device,optimizer,criterion) #Training
testing(net,"./MNISTmodelEpoch3iter10000",device,testloader,criterion) #Prediction