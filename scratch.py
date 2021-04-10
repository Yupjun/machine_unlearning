import torch
from torch.nn import functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from torch.utils.tensorboard import SummaryWriter


os.environ["CUBLAS_WORKSPACE_CONFIG"] =":16:8"



OBJECT_DATASET, test_dataset = None, None

# Reproduct configuration
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print('DEVICE : {}'.format(device))

parser = argparse.ArgumentParser(description='training')
parser.add_argument('--dataset', type=str,
                    help='setting dataset', default="mnist")

parser.add_argument('--epoch', type=int,
                    help='epoch of the model', default=100)

parser.add_argument('--batch_size', type=int,
                    help='Batch size in training', default=128)

parser.add_argument('--learning_rate', type=float,
                    help='learning rate of the model', default=0.001)

# baseline, scratch, KD
parser.add_argument('--phase', type=str,
                    help='phase of training baseline,scratch,KD', default="baseline")




args = parser.parse_args()

## configuration checkup print line

print("Current dataset  is %s" % args.dataset)
print("Current epoch  is %d" % args.epoch)
print("Current batch_size  is %d" % args.batch_size)
print("Current learning_rate  is %f" % args.learning_rate)
print("Current phase  is %s" % args.phase)
print('DEVICE : {}'.format(device))

# generate folder hierarchy
"""
FILE CONFIG
dataset
   baseline
     weight
     log
     result
   kd_output
     weight
     log
     result
   scratch
     weight
     log
     result
"""
cwd = os.getcwd()
PATH = cwd+"/" +args.dataset
path = cwd+"/" +args.dataset
GLOBAL_PATH = cwd+"/" +args.dataset+"/"+args.phase
os.makedirs(PATH, exist_ok=True)
os.makedirs(GLOBAL_PATH, exist_ok=True)
GLOBAL_PATH= cwd+"/" +args.dataset+"/"+args.phase+"/"

log_path= GLOBAL_PATH+"/"+"log"
result_path= GLOBAL_PATH+"/"+"result"
weight_path = GLOBAL_PATH+"/"+"weight"

os.makedirs(log_path, exist_ok=True)
os.makedirs(weight_path, exist_ok=True)
os.makedirs(result_path, exist_ok=True)

writer = SummaryWriter(log_path)


num_epochs = args.epoch
num_classes = 10 if args.dataset != 'cifar100' else 100
batch_size = args.batch_size
learning_rate = args.learning_rate

def dataset_splitter(DATASET, VALIDATION_SIZE, BATCH_SIZE):
    dataset_len = len(DATASET) - VALIDATION_SIZE
    validation_set, train_set  = torch.utils.data.random_split(DATASET, [VALIDATION_SIZE,dataset_len])
    TRAIN_DATALOADER = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    VALIDATION_DATALOADER = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True)

    print("TOTAL DATASET LENGTH")
    print(len(DATASET))
    print("TRANING DATASET length")
    print(len(train_set))
    print("validation set")
    print(len(validation_set))
    return TRAIN_DATALOADER, VALIDATION_DATALOADER


# dataset, model, optimizer prep

if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
    # Convolutional neural network (two convolutional layers)
    class Net(nn.Module):
        def __init__(self, num_classes=10):
            super(Net, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.fc = nn.Linear(7 * 7 * 32, num_classes)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            return out

elif args.dataset == 'cifar100' or args.dataset == 'SVHN':
    class Net(nn.Module):
        def __init__(self, my_pretrained_model):
            super(Net, self).__init__()
            self.pretrained = my_pretrained_model
            self.my_new_layers = nn.Linear(1000, num_classes)

        def forward(self, x):
            x = self.pretrained(x)
            x = self.my_new_layers(x)

            return x

elif args.dataset == 'cifar10':
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

### dataset generation

if args.dataset == 'cifar100':
    OBJECT_DATASET = torchvision.datasets.CIFAR100(root=path,
                                                   train=True,
                                                   transform=transforms.Compose(
                                                       [transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                                   download=True)
    test_dataset = torchvision.datasets.CIFAR100(root=path,
                                                 train=False,
                                                 transform=transforms.Compose(
                                                     [transforms.ToTensor(),
                                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                                 ,download=True)

elif args.dataset == 'cifar10':
    OBJECT_DATASET = torchvision.datasets.CIFAR10(root=path,
                                                  train=True,
                                                  transform=transforms.Compose(
                                                      [transforms.ToTensor(),
                                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                                  download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=path,
                                                train=False,
                                                transform=transforms.Compose(
                                                    [transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

elif args.dataset == 'SVHN':
    OBJECT_DATASET = torchvision.datasets.SVHN(root=path,
                                               split="train",
                                               transform=transforms.Compose(
                                                   [transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                               download=True)
    test_dataset = torchvision.datasets.SVHN(root=path,
                                             split="test",
                                             transform=transforms.Compose(
                                                 [transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                             , download=True)

elif args.dataset == 'fashion_mnist':
    OBJECT_DATASET = torchvision.datasets.FashionMNIST(root=path,
                                                       train=True,
                                                       transform=transforms.ToTensor(),
                                                       download=True)
    test_dataset = torchvision.datasets.FashionMNIST(root=path,
                                                     train=False,
                                                     transform=transforms.ToTensor(),
                                                     download=True)

elif args.dataset == 'mnist':
    print("test")
    OBJECT_DATASET = torchvision.datasets.MNIST(root='MNIST_data/',
                                                train=True,
                                                transform=transforms.ToTensor(),
                                                download=True)
    test_dataset = torchvision.datasets.MNIST(root='MNIST_data/',
                                              train=False,
                                              transform=transforms.ToTensor(),
                                              download=True)

else:
    print("something wrong %s" % args.dataset)


VALIDATION_SIZE = int(len(OBJECT_DATASET)*0.1)
train_dataloader,validation_dataloader= dataset_splitter(OBJECT_DATASET,VALIDATION_SIZE,batch_size)

model = None
if args.dataset == 'cifar100' or args.dataset == 'SVHN':
    resnet50 = torchvision.models.resnet50(pretrained=False, progress=True)
    resnet50.to(device)
    resnet50.train()
    model = Net(my_pretrained_model=resnet50).to(device)
else:
    model = Net().to(device)

print(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Train the model
total_step = len(train_dataloader)

GLOBAL_EPOCH=0
GLOBAL_ACCURACY=0

for epoch in range(num_epochs):
    """
    non_alpha_dataloder와 alpha_dataloder를 합치면 우리가 학습해야하는 total dataset이 됩니다. 
    random subset 다시 합치는 방법을 몰라서.. 일단 이렇게 구현했는데 별문제 없을거같아요
    """
    model.train()
    for i, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            writer.add_scalar('training loss',
                              loss.item(),
                              epoch * len(train_dataloader) + i)

    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in validation_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
        acc= 100 * correct / total
        writer.add_scalar('validation accruacy',
                          acc,
                          epoch )


    # Save the model checkpoint
    file_name = args.dataset+"_"+str(epoch+1)+".pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
    os.path.join(weight_path,file_name))
    if acc>GLOBAL_ACCURACY:
    	GLOBAL_EPOCH= epoch+1
    	GLOBAL_ACCURACY = acc

print("Total output: GLOBAL_ACCURACY is %f and GLOBAL_EPOCH is %d" % (GLOBAL_ACCURACY,GLOBAL_EPOCH))
name = args.dataset+".txt"
content = "Total output: GLOBAL_ACCURACY is"+str(GLOBAL_ACCURACY)+"and GLOBAL_EPOCH is "+str(GLOBAL_EPOCH)
with open(name, "w") as f:
    f.write(content)



