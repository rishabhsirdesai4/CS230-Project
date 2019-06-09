#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import h5py
import math
import scipy
import torch
import pdb
import utils
import torch.nn as nn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models


# In[2]:


os.sys.path


# In[3]:


home = os.path.expanduser("~")
dataRoot = os.path.join(home, "CS230", "CheXpert-v1.0-small")
os.listdir(dataRoot)


# In[4]:


"""Import and Preprocess Data"""


# In[5]:


import cv2

TASKS = ['No Finding', 'Enlarged Cardiomediastinum', 
         'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 
         'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
        'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
        'Fracture', 'Support Devices']

def view_dataset(paths, labels, method='cv2'):
    fig, axs = plt.subplots(2, 7, figsize=(10, 10))
    flatted_axs = [item for one_ax in axs for item in one_ax]
    for ax, path, label in zip(flatted_axs, paths[:14], labels[:14]):
        if method == 'cv2':
            img = cv2.imread('/home/ubuntu/CS230/' + path, 3)
        elif method == 'tf':
            img = try_tf_image(path)
        ax.imshow(img)
        ax.set_title(label)
        ax.axis('off')
    plt.show() 

from torch.nn import functional as F

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

df = pd.read_csv('/home/ubuntu/CS230/CheXpert-v1.0-small/train_final_V2.csv')
paths = df['Path'][:14]
labels = df[TASKS][:14]

view_dataset(paths, labels)


# In[6]:


from torch.utils.data import Dataset, DataLoader

class Dataset(Dataset):
    def __init__(self, data_split, toy=False):
     
        df = pd.read_csv(data_split)
        
        # Remove any paths for which the image files do not exist
        #df = df[df["Path"].apply(os.path.exists)]
      
        #print ("%s size %d" % (data_split, df.shape[0]))

        #Could remove
        if toy:
            df = df.sample(frac=0.05)

        self.img_paths = df["Path"].apply(lambda x: '/home/ubuntu/CS230/' + x)
        self.labels = df[TASKS]
        self.n_classes = len(self.labels)
        self.transforms = transforms.Compose([
                            transforms.CenterCrop((224,224)),
                            transforms.ToTensor(),
                        ])
                        

    def __getitem__(self, index):
        img = Image.open(self.img_paths.iloc[index]).convert("RGB")
        img = self.transforms(img)
        label = self.labels.iloc[index]
        label = np.nan_to_num(label)
        label[label < 0] = 0
        label_vec = torch.FloatTensor(label).cuda()
        return img, label_vec
          

    def __len__(self):
        return len(self.img_paths)


# In[7]:


train_dataset = Dataset('/home/ubuntu/CS230/CheXpert-v1.0-small/train_final_V2.csv',toy=False)
test_dataset = Dataset('/home/ubuntu/CS230/CheXpert-v1.0-small/valid_final.csv',toy=False)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                         shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4,
                                         shuffle=False)

print(len(train_loader))
print(len(test_loader))
print(train_loader)


# In[8]:


#Define Model


# In[9]:




class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, growth_rate=32,
                 num_init_features=224, bn_size=4, drop_rate=0, num_classes=14):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


# In[10]:


#densenet = models.densenet161(pretrained=True)


# In[11]:


import torch.optim as optim

model = DenseNet121().cuda()
# Define the cost function
criterion = nn.BCEWithLogitsLoss().cuda()

# Define the optimizer, learning rate 
lr = 0.00001
optimizer = optim.Adam(model.parameters(), lr)


# In[12]:


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


# In[26]:



from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

#Train Network
total_step = len(train_loader)
epoch_list = []
loss_list = []
acc_list = []
total_recall = []
f1_total = []


best_val_acc = 0.0

for epoch in range(3):
    #logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs = inputs.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward propogation
        outputs = model(inputs)
        outputs = outputs.cuda()
        
        # calculate the loss
        loss = criterion(outputs, labels)
        
        # backpropogation + update parameters
        loss.backward()
        optimizer.step()

        # print statistics
        cost = loss.item()
        state = {'epoch': epoch + 1,
         'state_dict': model.state_dict(),
         'optim_dict' : optimizer.state_dict()}
        
        torch.save(model.state_dict(), '/home/ubuntu/CS230/CS230-Project/Model/result_densenet.pt')
        
#         utils.save_checkpoint({'epoch': epoch + 1,
#                                'state_dict': model.state_dict(),
#                                'optim_dict' : optimizer.state_dict()},
#                                is_best=is_best,
#                                checkpoint=model_dir)
        
        outputs = torch.sigmoid(outputs).cpu().detach().numpy()
        outputs[outputs >= 0.5] = 1
        outputs[outputs < 0.5] = 0
        labels = labels.cpu().detach().numpy() 
        
        recall = recall_score(labels,outputs, average = 'micro')
        total_recall.append(recall)
        
        f1 = f1_score(labels,outputs, average = 'micro')
        f1_total.append(f1)

        if i % 100 == 0:    # print every 1000 iterations
            print('Epoch:' + str(epoch) + ", Iteration: " + str(i) 
                  + ", training cost = " + str(cost) + ", Recall score =" + str(recall) 
                  + ", F1 score =" + str(f1))
            loss_list.append(cost)
            epoch_list.append(epoch*12500 + i)

plt.plot(epoch_list,loss_list)
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate =" + str(lr))
plt.show()

print("Average recall =" + str(sum(total_recall)/len(total_recall)))
print("Average F1 score is: ",sum(f1_total)/len(f1_total))


# In[28]:


print("Average F1 score is: ",sum(f1_total)/len(f1_total))


# In[17]:



from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from torch.autograd import Variable

def test():
    model = DenseNet121().cuda()
    model.load_state_dict(torch.load('/home/ubuntu/CS230/CS230-Project/Model/result_densenet.pt'))
    test_recall = []
    f1_test = []
    model.eval()
    test_acc = 0.0
    for i, (images, labels) in enumerate(test_loader):

        
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

        # Predict classes using images from the test set
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        
        outputs = torch.sigmoid(outputs).cpu().detach().numpy()
        outputs[outputs >= 0.5] = 1
        outputs[outputs < 0.5] = 0
        labels = labels.cpu().detach().numpy()
        
        #test_acc += torch.sum(prediction == labels.data)
        
        recall = recall_score(labels,outputs, average = 'micro')
        test_recall.append(recall)
        
        f1 = f1_score(labels,outputs, average = 'micro')
        f1_test.append(f1)

    # Compute the average acc and loss over all 10000 test images
    #test_acc = test_acc / 10000
    
    print("Average recall =" + str(sum(test_recall)/len(test_recall)))
    print("Average F1 score is: ",sum(f1_test)/len(f1_test))

    return test_acc


# In[18]:


test()


# In[ ]:




