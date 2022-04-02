import torch
from torch.nn import functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import math
import argparse
from torch.utils.tensorboard import SummaryWriter
import copy

# GPU memory setting
os.environ["CUBLAS_WORKSPACE_CONFIG"] =":16:8"
OBJECT_DATASET, test_dataset = None, None

# GPU checking 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('DEVICE : {}'.format(device))

# Reproduction configuration
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)


kd_alpha =0.9

# argparse 

# example 
# testing.py --dataset mnist --epoch 50 --batch_size 128 --learning_rate 0.001 --best_epoch 24 --phase experiment

parser = argparse.ArgumentParser(description='training')

parser.add_argument('--dataset', type=str,
                    help='setting dataset', default="mnist")

parser.add_argument('--epoch', type=int,
                    help='epoch of the model', default=100)

parser.add_argument('--batch_size', type=int,
                    help='Batch size in training', default=128)

parser.add_argument('--learning_rate', type=float,
                    help='learning rate of the model', default=0.001)
parser.add_argument('--best_epoch', type=int,
                    help='learning rate of the model', default=20)


parser.add_argument('--phase', type=str,
                    help='phase of training baseline,experiment', default="experiment")



class CustumDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.targets = target
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = torch.tensor(self.data[idx])
        if self.transform:
            img = self.transform(self.data[idx])
        return img, self.targets[idx]

def dataset_splitter(DATASET, SPLIT_VALUE, VALIDATION_SIZE, BATCH_SIZE, specific_alpha = False):
    #available length of dataset
    dataset_len = len(DATASET) - VALIDATION_SIZE
    # validation for validate
    # non alpha is pure data that can be utilized for neutralization and scratch
    # alpha is for neutralization
    validation_set, non_alpha_set, alphaset_set = torch.utils.data.random_split(DATASET, [VALIDATION_SIZE,
                                                                                          dataset_len - SPLIT_VALUE,
                                                                                          SPLIT_VALUE])
    ALPHA_DATALOADER = None
    NON_ALPHA_DATALOADER = torch.utils.data.DataLoader(non_alpha_set, batch_size=BATCH_SIZE, shuffle=True)
    alphaset_set_custum=None



    if specific_alpha:
        mask = np.array(DATASET.targets) != 4
        list_temp_specific_idx = []
        print(len(alphaset_set.indices))
        for i in alphaset_set.indices :
            if mask[i] : #true이면
                list_temp_specific_idx.append(i)
        print(len(list_temp_specific_idx))

        label_out_dataset = alphaset_set.dataset.data[list_temp_specific_idx]
        label_out_targets = np.array(alphaset_set.dataset.targets)[list_temp_specific_idx]
        transform_train = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        alphaset_set_custum = CustumDataset(label_out_dataset, label_out_targets, copy.deepcopy(transform_train))
        ALPHA_DATALOADER = torch.utils.data.DataLoader(alphaset_set_custum, batch_size=BATCH_SIZE, shuffle=True)
    else :
        if SPLIT_VALUE>32:
            ALPHA_DATALOADER_NEUT = torch.utils.data.DataLoader(alphaset_set, batch_size=1, shuffle=True)
            ALPHA_DATALOADER = torch.utils.data.DataLoader(alphaset_set, batch_size=BATCH_SIZE, shuffle=True)
        else:
            ALPHA_DATALOADER_NEUT = torch.utils.data.DataLoader(alphaset_set, batch_size=1, shuffle=True)
            ALPHA_DATALOADER = torch.utils.data.DataLoader(alphaset_set, batch_size=BATCH_SIZE, shuffle=True)
    VALIDATION_DATALOADER = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True)

    print("TOTAL DATASET LENGTH")
    print(len(DATASET))
    print("non alpha length")
    print(len(non_alpha_set))
    print("alpha length")
    if alphaset_set_custum :
        print(len(alphaset_set_custum))
    else:
        print(len(alphaset_set))
    print("validation set")
    print(len(validation_set))
    return NON_ALPHA_DATALOADER, ALPHA_DATALOADER, VALIDATION_DATALOADER, ALPHA_DATALOADER_NEUT

def loss_fn_kd(outputs, labels, teacher_outputs,kd_alpha=0.9,kd_t=20):
    KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/kd_t,dim=1),
                             F.softmax(teacher_outputs/kd_t,dim=1) * kd_alpha*kd_t*kd_t) +\
        F.cross_entropy(outputs, labels) * (1. - kd_alpha)
    return KD_loss

def targets_randomizing(range_label,correct_targets): #ex) 클래스갯수, trueLabels로 구성
    result = [x + 1 if x + 1 != range_label else 0 for x in correct_targets]
    return result

def Get_List_contrasive_index(targets,targets_mask):
    _targets = np.array(targets)
    _targets_mask = np.array(targets_mask)
    list_masked = _targets_mask[_targets]
    # print(list_masked.tolist())
    # print(_targets)
    return list_masked.tolist()


args = parser.parse_args()

## configuration checkup print line

print("Current dataset  is %s" % args.dataset)
print("Current epoch  is %d" % args.epoch)
print("Current batch_size  is %d" % args.batch_size)
print("Current learning_rate  is %f" % args.learning_rate)
print("Current phase  is %s" % args.phase)
print('DEVICE : {}'.format(device))

# generate folder
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

weight_path = GLOBAL_PATH+"/"+"weight"
os.makedirs(weight_path, exist_ok=True)

num_epochs = args.epoch
num_classes = 10 if args.dataset != 'cifar100' else 100
print("num_classes is %d" % num_classes )
batch_size = args.batch_size
learning_rate = args.learning_rate

if args.phase =="baseline":
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

model = None


if args.dataset == 'cifar100' or args.dataset == 'SVHN':
    resnet50 = torchvision.models.resnet50(pretrained=False, progress=True)
    resnet50.to(device)
    resnet50.train()
    model = Net(my_pretrained_model=resnet50).to(device)
else:
    model = Net().to(device)



#start of retraning whole 
AVAILABLE_LENGTH = len(OBJECT_DATASET)-VALIDATION_SIZE

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)



for alpha_ratio in np.arange(0.1,10.1,0.1):
    alpha_ratio =np.around(alpha_ratio,2)
    name = cwd+"/"+args.dataset+"/baseline/weight/"+args.dataset+ "_"+str(args.best_epoch)+".pth"
    checkpoint = torch.load(name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()



    loss_CE = torch.nn.CrossEntropyLoss()
    selected_model=model
    selected_model.eval()

    # get the alpha value to extract
    SPLIT_VALUE = int(AVAILABLE_LENGTH*alpha_ratio/100)
    """
    non_alpha_dataloader -> dataset without amount of alpha / valiation / test [for scratch]
    alpha_dataloader     -> dataset that consist of alpha [for neutralization]
    validation_dataloader-> literally validation dataset [for validation]
    neut_dataloader      -> dataset for retraining the neutralized model(without alpha,) [for retraining]
    """
    non_alpha_dataloader,alpha_dataloader,validation_dataloader,neut_dataloader= dataset_splitter(OBJECT_DATASET,SPLIT_VALUE,VALIDATION_SIZE,batch_size,False)
    
    # Rebalance the class label to far

    old_label_list = []
    # numclass 만큼의 list를 만들고
    # 각 list는 numclass만큼의 element가 있음
    LABEL_MAPPER = [0 for x in range(num_classes)]
    final_label_list = [x for x in range(num_classes)]
    with torch.no_grad():
        for _, (img, label) in enumerate(neut_dataloader):
            output = selected_model(img.to(device))
            old_label_list.append(label[0].item())
            LABEL_MAPPER[label] += output.cpu()

    # alpha 안에서 존재하는 class만 뽑아낸
    LABEL_MAP_LIST = set(old_label_list)
    print("LABEL_MAP_LIST:available class list")
    print(LABEL_MAP_LIST)

    # 차집합
    INIT_SET = set([x for x in range(num_classes)])
    DIFFERENCE_LIST = list(INIT_SET.difference(LABEL_MAP_LIST))

    for num in range(num_classes):
        if num in DIFFERENCE_LIST: continue
        LABEL_MAPPER[num][0] /= old_label_list.count(num)

    for del_element in DIFFERENCE_LIST:
        LABEL_MAPPER[del_element] = torch.tensor([[float('inf')]*num_classes])
    # erase cannot use class

    # erase unit matrix element
    for index in range(num_classes):
        LABEL_MAPPER[index][0][index] = float('inf')

    prev_from_class, prev_to_class =0,0
    for index,_ in enumerate(list(LABEL_MAP_LIST)):
        MINIMUM_FINDER = [min(LABEL_MAPPER[x][0]) for x in range(num_classes)]
        from_class=MINIMUM_FINDER.index(min(MINIMUM_FINDER))##########
        #if float('inf') not in LABEL_MAPPER[from_class]:
        to_class = torch.argmin(LABEL_MAPPER[from_class][0].clone().detach().requires_grad_(True))########
        print('from_class , to_class : {}, {}'.format(from_class, to_class))
        
        final_label_list[from_class] = int(to_class)
        # erase minimum_value
        
        for index in range(num_classes):
            LABEL_MAPPER[index][0][to_class] = float('inf')
        LABEL_MAPPER[from_class]=[[float('inf')]*num_classes]

        bCheck_values=False
        for l_final in LABEL_MAPPER:
            for l in l_final[0]:
                if not math.isinf(l):
                    bCheck_values = True
                    print("not isinf")
                    break
            if bCheck_values:break
        if not bCheck_values:
            for id_check in range(len(final_label_list)):
                if final_label_list[id_check] == id_check:
                    final_label_list[id_check] = to_class.item()
                    final_label_list[from_class] = id_check
            break
        prev_from_class = from_class
        prev_to_class = to_class

    print("DIFFERENCE_LIST")
    print(DIFFERENCE_LIST)
    print("LABEL_MAP_LIST")
    print(LABEL_MAP_LIST)
    print("final_label_list")
    print(final_label_list)


    # generate folder
    """
    FILE CONFIGURATION

    dataset
       baseline
         weight
         log
       experiment -> GLOBAL PATH
           kd_output 
             weight 
               0.1[value]
             log
               0.1[value]
           scratch
             weight
               0.1[value]
             log
               0.1[value]
    """
    
    cwd = os.getcwd()
    PATH = cwd+"/" +args.dataset
    os.makedirs(PATH, exist_ok=True)

    # experiment folder
    GLOBAL_PATH = cwd+"/" +args.dataset+"/"+args.phase
    os.makedirs(GLOBAL_PATH, exist_ok=True)

    # kd_output folder
    kd_output_path= GLOBAL_PATH+"/"+"kd_output"
    kd_weight_path= kd_output_path + "/"+"weight"
    kd_weight_value_path= kd_weight_path+"/"+str(alpha_ratio)

    kd_log_path= kd_output_path+"/"+"log"
    kd_log_value_path = kd_log_path +"/"+str(alpha_ratio)


    os.makedirs(kd_output_path, exist_ok=True)
    os.makedirs(kd_weight_path, exist_ok=True)
    os.makedirs(kd_weight_value_path, exist_ok=True)
    os.makedirs(kd_log_path, exist_ok=True)
    os.makedirs(kd_log_value_path, exist_ok=True)


    # scratch folder 
    scratch_output_path= GLOBAL_PATH+"/"+"scratch"
    scratch_weight_path= scratch_output_path + "/"+"weight"
    scratch_weight_value_path= scratch_weight_path+"/"+str(alpha_ratio)
    scratch_log_path= scratch_output_path+"/"+"log"
    scratch_log_value_path = scratch_log_path +"/"+str(alpha_ratio)



    os.makedirs(scratch_output_path, exist_ok=True)
    os.makedirs(scratch_weight_path, exist_ok=True)
    os.makedirs(scratch_weight_value_path, exist_ok=True)
    os.makedirs(scratch_log_path, exist_ok=True)
    os.makedirs(scratch_log_value_path, exist_ok=True)

    #tensorboard logging for both training part
    #kd_writer log neutralization part
    kd_writer = SummaryWriter(kd_log_value_path)



    
    if args.dataset == 'cifar100' or args.dataset == 'SVHN':
        selected_model.pretrained.train()
    
    NEUTRAL_EPOCH = 500000
    
    optimizer = torch.optim.Adam(selected_model.parameters(), lr=learning_rate)
    NEUT_THREASHOLD =1/num_classes 
    for epoch in range(NEUTRAL_EPOCH):
        selected_model.train()
        if args.dataset == 'cifar100' or args.dataset == 'SVHN':
            selected_model.pretrained.train()

        for i, (images, labels) in enumerate(alpha_dataloader):

            labels = Get_List_contrasive_index(labels,final_label_list)
            images = images.to(device)
            labels = torch.tensor(labels).to(device)
            outputs = selected_model(images)
            loss = loss_CE(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        selected_model.eval()
        if args.dataset == 'cifar100' or args.dataset == 'SVHN':
            selected_model.pretrained.eval()

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in alpha_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = selected_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print((100 * correct / total))
            kd_writer.add_scalar('neutralization log acc on alpha',100 * correct / total,epoch+1)

        if correct / total < NEUT_THREASHOLD:
            print('Neutralized percentage: {}% | epoch : {} '.format(100 * correct / total, epoch))
            kd_writer.add_scalar('stopped epoch in neutralization',100 * correct / total,1)
            file_name = kd_output_path+"/"+"neutralization_output.pth"
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': selected_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},file_name)
            print("neutralization success it is saved at")
            print(file_name)
            break
    
    # test output with neutralized part
    selected_model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = selected_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the selected_model after Neutralize: {} %'.format(100 * correct / total))
        kd_writer.add_scalar('test accuracy of neutralized model',100 * correct / total,1)

    # model & dataset part
    teacher_model = None

    if args.dataset == 'cifar100' or args.dataset == 'SVHN':
        resnet50_4 = torchvision.models.resnet50(pretrained=False, progress=True)
        resnet50_4.to(device)
        teacher_model = Net(my_pretrained_model=resnet50_4).to(device)
    elif args.dataset == 'cifar10':
        teacher_model = Net().to(device)
    elif args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
        teacher_model = Net(num_classes).to(device)

    # should be changed -> 
    # total_model_location
    name = cwd+"/"+args.dataset+"/baseline/weight/"+args.dataset+"_"+str(args.best_epoch)+".pth"
    
    checkpoint = torch.load(name)
    teacher_model.load_state_dict(checkpoint['model_state_dict'])
    teacher_model.eval()
    if args.dataset == 'cifar100' or args.dataset == 'SVHN':
        teacher_model.pretrained.eval()
    student_model = copy.deepcopy(selected_model)


    kd_lr = 0.001
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=kd_lr)
    num_epochs_KD = 50
    num_withKDLoss_epochs= 10
    best_acc = 0
    best_epoch_kd=0
    for epoch in range(num_epochs_KD):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, num_epochs_KD, kd_lr))
        # train
        teacher_model.eval()
        if args.dataset == 'cifar100' or args.dataset == 'SVHN':
            teacher_model.pretrained.eval()

        student_model.train()
        if args.dataset == 'cifar100' or args.dataset == 'SVHN':
            student_model.pretrained.train()
        correct, total = 0, 0
        teacher_outputs, loss_kd, loss = None, 0, 0
        
        for batch_idx, (inputs, targets) in enumerate(non_alpha_dataloader):
            # use almost half of the dataset
            if batch_idx%2 ==0:
                pass
            else:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = student_model(inputs)
                loss_main = criterion(outputs, targets)
                if epoch < num_withKDLoss_epochs:
                    teacher_outputs = teacher_model(inputs)
                    loss_kd = loss_fn_kd(outputs, targets, teacher_outputs, kd_alpha = kd_alpha)
                    loss = loss_main + loss_kd
                else:
                    loss = loss_main
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    class_correct = list(0. for i in range(10))
                    class_total = list(0. for i in range(10))
                    _, predicted = torch.max(outputs, 1)  # mhkim
                    c = (predicted == targets).squeeze()
                    correct += (predicted == targets).sum().item()
                    total += len(targets)
        
        print('non_alpha_dataloader ACC : {acc:.2f}'.format(acc=correct / total * 100))
        kd_writer.add_scalar('train data accuracy in retraining',100 * correct / total,epoch+1)
        # validation
        correct, total = 0, 0
        student_model.eval()
        if args.dataset == 'cifar100' or args.dataset == 'SVHN':
            student_model.pretrained.eval()
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(validation_dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = student_model(inputs)
                loss = criterion(outputs, targets)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == targets).squeeze()
                correct += (predicted == targets).sum().item()
                total += len(targets)
        
        print('validation_dataloader ACC : {acc:.2f}'.format(acc=correct / total * 100))
        kd_writer.add_scalar('retraining validation accuracy',100 * correct / total,epoch+1)
        if correct / total > best_acc:
            best_epoch_kd = epoch+1
            best_acc = correct / total
            file_name = kd_weight_value_path+"/"+args.dataset+".pth"
            torch.save({
              'epoch': epoch+1,
              'model_state_dict': student_model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict()},file_name)

    final_model = None
    if args.dataset == 'cifar100' or args.dataset == 'SVHN':
        resnet50_5 = torchvision.models.resnet50(pretrained=False, progress=True)
        resnet50_5.to(device)
        final_model = Net(my_pretrained_model=resnet50_5).to(device)
    elif args.dataset == 'cifar10':
        final_model = Net().to(device)
    elif args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
        final_model = Net(num_classes).to(device)

    # Reload best accuracy model
    print(best_epoch_kd)
    name = kd_weight_value_path+"/"+args.dataset+".pth"
    print(name)
    checkpoint = torch.load(name)
    final_model.load_state_dict(checkpoint['model_state_dict'])
    final_model.eval()
    
    # caluclate accuracy on test dataset
    if args.dataset == 'cifar100' or args.dataset == 'SVHN':
        final_model.pretrained.eval()
    
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = final_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the final after KD: {} %'.format(100 * correct / total))
        kd_writer.add_scalar('final test accuracy of retrained model',100 * correct / total,1)
    

    #scratch model training 
    without_alpha_model = None
    resnet50 = None
    
    if args.dataset == 'cifar100' or args.dataset == 'SVHN':
        resnet50_6 = torchvision.models.resnet50(pretrained=False, progress=True)
        resnet50_6.to(device)
        without_alpha_model = Net(my_pretrained_model=resnet50_6).to(device)
    elif args.dataset == 'cifar10':
        without_alpha_model = Net().to(device)
    elif args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
        without_alpha_model = Net(num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(without_alpha_model.parameters(), lr=learning_rate)
    best_acc = 0
    BEST_WITHOUT_ALPHA_EPOCH = 0
    BEST_WITHOUT_ALPHA_ACCURACY = 0
    # Train the model
    total_step = len(non_alpha_dataloader)
    loss_index=1
    num_withKDLoss_epochs= 10
    num_epochs_WITHOUT= 50
    for epoch in range(num_epochs_WITHOUT):
        without_alpha_model.train()
        if args.dataset == 'cifar100' or args.dataset == 'SVHN':
            without_alpha_model.pretrained.train()
        for i, (images, labels) in enumerate(non_alpha_dataloader):
            if i%2 ==0:
                pass
            else:        
              images = images.to(device)
              labels = labels.to(device)
  
              # Forward pass
              outputs = without_alpha_model(images)
              loss_main =criterion(outputs,labels)
              if epoch < num_withKDLoss_epochs:
                  teacher_outputs = teacher_model(images)
                  loss_kd = loss_fn_kd(outputs, labels, teacher_outputs, kd_alpha = kd_alpha)
                  loss = loss_main + loss_kd
              else:
                  loss = loss_main
  
              # Backward and optimize
              optimizer.zero_grad()
              loss.backward(retain_graph=True)
              optimizer.step()
  
              if (i + 1) % 100 == 0:
                  print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                        .format(epoch + 1, num_epochs_WITHOUT, i + 1, total_step, loss.item()))
                  kd_writer.add_scalar('without scratch model loss',loss.item(),loss_index )
                  loss_index+=1
        
        without_alpha_model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        if args.dataset == 'cifar100' or args.dataset == 'SVHN':
            without_alpha_model.pretrained.eval()
        

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in validation_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = without_alpha_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('validation Accuracy of the without_alpha model on the  test images: {} %'.format(100 * correct / total))
            kd_writer.add_scalar('validation Accuracy of scratch model',100 * correct / total,epoch+1 )
        if 100 * correct / total > BEST_WITHOUT_ALPHA_ACCURACY:
            BEST_WITHOUT_ALPHA_ACCURACY = 100 * correct / total
            BEST_WITHOUT_ALPHA_EPOCH = epoch + 1
            # this should be changed
            file_name = scratch_weight_value_path+"/"+args.dataset+"_"+".pth"
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},file_name)

    print("best without alpha epoch: %d" % BEST_WITHOUT_ALPHA_EPOCH)
    print("best without alpha accuracy: %f " % BEST_WITHOUT_ALPHA_ACCURACY)


    final_model=None
    if args.dataset == 'cifar100' or args.dataset == 'SVHN':
        resnet50_7 = torchvision.models.resnet50(pretrained=False, progress=True)
        resnet50_7.to(device)
        final_model = Net(my_pretrained_model=resnet50_7).to(device)
    elif args.dataset == 'cifar10':
        final_model = Net().to(device)
    elif args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
        final_model = Net(num_classes).to(device)

    checkpoint = None
    file_name = scratch_weight_value_path+"/"+args.dataset+"_"+".pth"
    checkpoint = torch.load(file_name)
    final_model.load_state_dict(checkpoint['model_state_dict'])
    final_model.eval()

    if args.dataset == 'cifar100' or args.dataset == 'SVHN':
        final_model.pretrained.eval()
    
    # log test data
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = without_alpha_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the final after Scratch: {} %'.format(100 * correct / total))
        kd_writer.add_scalar('est Accuracy of the final after Scratch',100 * correct / total,1)