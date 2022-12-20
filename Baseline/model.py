import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm
from torchvision import transforms, models
import numpy
from transformers import CLIPProcessor, CLIPVisionModel
class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fc0 = (nn.Linear(10,2352))
        self.fc1 = (nn.Linear(10, 512))
        self.fc2 = (nn.Linear(512, 1024))
        self.fc3 = (nn.Linear(1024, 2048))
        self.fc4 = (nn.Linear(2048, 4096))
        self.fc5 = (nn.Linear(300,8192))
        self.fc6 = (nn.Linear(5,64))
        self.fc7 = (nn.Linear(64,5))
        self.fc8 = (nn.Linear(125,64))
        self.fc9 = (nn.Linear(64,125))
        self.conv1 = (nn.Conv2d(16, 32, 3, 1, 1))
        self.conv2 = (nn.Conv2d(32, 32, 3, 1, 1))
        self.conv3 = (nn.Conv2d(8, 16, 3, 1, 1))
        self.conv4 = (nn.Conv2d(16, 16, 3, 1, 1))
        self.conv5 = (nn.Conv2d(4, 8, 3, 1, 1))
        self.conv6 = (nn.Conv2d(8, 4, 3, 1, 1))
        self.pixel_shuffle = nn.PixelShuffle(2)
        

        self.model1 = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor1 = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, x):
#         a=0
#         if x.shape[0]==5:
#             a=1
#             print(x.shape)
#             x = x.contiguous().view(-1,5)
#             x = F.relu(self.fc6(x))
#             x = x.view(64,-1)
#         if x.shape[0]==125:
#             a=2
#             print(x.shape)
#             x = x.contiguous().view(-1,125)
#             x = F.relu(self.fc8(x))
#             x = x.view(64,-1)
            
        
#         x1 = F.relu(self.fc0(x))
        
#         x1 = x1.view(3,224,224)
#         x1 = x1.detach().cpu()
#         inputs = self.processor1(images=x1, return_tensors="pt")
#         inputs.to(self.device)
        
#         outputs = self.model1(**inputs)
#         x2 = outputs.last_hidden_state
#         x2 = x2.view(128,300)
#         x2 =  F.relu(self.fc5(x2))
#         x2 = x2.view(64,128,128)
#         if a==1:
#             x2 = x2.view(16384,64)
#             x2 = F.relu(self.fc7(x2))
#             x2 = x2.view(5,128,128)
            
#         if a==2:
#             x2 = x2.view(16384,64)
#             x2 = F.relu(self.fc9(x2))
#             x2 = x2.view(125,128,128)
            
#         print(x2.size())
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 16, 16, 16)
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pixel_shuffle(self.conv6(x))
        x = torch.sigmoid(x)
#         return x2
        return 1 - x.view(-1, 128, 128)
