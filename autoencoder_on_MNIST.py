"""
Author              : Krupal Shah
Adapted from        : https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/simple_autoencoder.py
Program Description : Plain autoencoder to reconstruct MNIST images.
                      Implemented checkpoint restart logic to save progress 
                      in case of unprecedented exit from the script
"""

import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from datetime import datetime

# generate folder for all files using current timestamp
folder = datetime.now().strftime("%d%b%Y-%H%M")

os.mkdir(folder)

if not os.path.exists('./' + folder + '/mlp_img'):
    os.mkdir('./' + folder + '/mlp_img')

#if not os.path.exists(folder + './mlp_img'):
#    os.mkdir(folder + './mlp_img')

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


start_epoch = 0
num_epochs = 3
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = MNIST('./data', transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

os.chdir(folder)


model = autoencoder().cpu()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
restart = False

# restore checkpoints while restarting the script   
if restart==True:
    restart_path = "PROVIDE RESTART PATH HERE"
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    num_epochs = checkpoint['num_epochs']
    loss = checkpoint['loss']
    os.chdir(checkpoint['current_directory'])
    print(os.getcwd())


for epoch in range(start_epoch, num_epochs):
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        img = Variable(img).cpu()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))
    
    # Visualisation of original and regenerated images
    #if epoch % 10 == 0:


    pic = to_img(img.cpu().data)
    save_image(pic, './mlp_img/original_{}.png'.format(epoch))

    pic = to_img(output.cpu().data)    
    save_image(pic, './mlp_img/regenerated_{}.png'.format(epoch))
    
    path = os.path.join( os.getcwd(), "MNIST_AE.pk")
    torch.save({
            'epoch': epoch+1,
            'num_epochs' : num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'current_directory':os.getcwd()
            }, path)

    # Artificial break to test the checkpoint-restart logic
    if epoch==1 and restart==False:
    	print("Breaking the loop in between")
    	print("Path :", path)
    	break
    	
print("End of the script")
