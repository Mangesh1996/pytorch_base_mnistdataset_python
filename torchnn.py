import torch
from torch import nn,save,load
from torchvision.transforms import ToTensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
import os
train=datasets.MNIST(root="data",download=True,train=True,transform=ToTensor())
dataset=DataLoader(train,32)
#Image classification and nerual network
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Conv2d(1,32,(3,3)),
            nn.ReLU(),
            nn.Conv2d(32,64,(3,3)),
            nn.ReLU(),
            nn.Conv2d(64,64,(3,3)),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6),10)
        )

    def forward(self,X):
        return self.model(X)

clf=ImageClassifier().to('cuda')
opt=Adam(clf.parameters(),lr=1e-3)
loss_fn=nn.CrossEntropyLoss()
#Traing flow

def train():
    for epoch in range(10):
        for batch in dataset:
            X,y=batch
            X,y=X.to("cuda"),y.to("cuda")
            yhat=clf(X)
            loss=loss_fn(yhat,y)
            #Apply backprop

            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Epoch  {epoch} loss is {loss.item()}")
    os.makedirs("model",exist_ok=True)
    with open("model/model_state.pt","wb") as f :
        save(clf.state_dict(),f)

if __name__=="__main__":
    train()