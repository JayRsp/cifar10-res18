import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import timm
import random
import numpy as np
from torch import multiprocessing   
from tqdm import tqdm

model_num = 2 
total_epoch, lr = 80, 1e-4

def run():
    multiprocessing.freeze_support()
    
    global model_num, total_epoch, lr
    
    for s in range(model_num):
        
        
        # fix random seed
        seed_number = s
        random.seed(seed_number)
        np.random.seed(seed_number)
        torch.manual_seed(seed_number)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'using device is: {device}')

        # Define the data transforms
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Load the CIFAR-10 dataset
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=12)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=12)

        # Define the ResNet-18 model with pre-trained weights
        model = timm.create_model('resnet18', pretrained=True, num_classes=10)
        model = model.to(device)  # Move the model to the GPU

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # Define the learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        def train():
            model.train()
            running_loss = 0.0
            
            train_iterator = tqdm(trainloader, total=len(trainloader))
            
            for i, data in enumerate(train_iterator, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)  # Move the input data to the GPU
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if i % 100 == 99:
                    train_iterator.set_description(f'Epoch {epoch + 1}, Loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

        def test():
            model.eval()
            
            correct = 0
            total = 0

            test_iterator = tqdm(testloader, total=len(testloader))

            with torch.no_grad():
                for data in test_iterator:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)  # Move the input data to the GPU
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    test_iterator.set_description(f'Accuracy: {100 * correct / total:.2f}%')

            print('Accuracy on the 10000 test images: %f %%' % (100 * correct / total))

        for epoch in range(total_epoch):
            train()
            test()
            scheduler.step()

        print('Finished Training')

        PATH = './resnet18_cifar10_%f_%d.pth' % (lr, seed_number)
        torch.save(model.state_dict(), PATH)

if __name__ == '__main__':
    run()
	