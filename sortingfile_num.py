from torch import nn,save,load
import glob
from PIL import Image
from torchvision.transforms import ToTensor
import shutil
from torch.optim import Adam
import os
import torch
from torchnn import ImageClassifier
import sys
try:
    clf=ImageClassifier().to('cuda')
    opt=Adam(clf.parameters(),lr=1e-3)
    loss_fn=nn.CrossEntropyLoss()

    with open("model/model_state.pt","rb") as read:
        clf.load_state_dict(load(read))
    images = glob.glob("data_sample/*.jpg")
    number_find=dict()
    for img in images:
        with open(img, 'rb') as file:
            img1 = Image.open(file)
            img_tensor=ToTensor()(img1).unsqueeze(0).to("cuda")
            key=int(str(torch.argmax(clf(img_tensor))).split(",")[0].split("(")[1])
            number_find.setdefault(key,[]).append(img)
        os.makedirs(os.path.join("sorting",str(key)),exist_ok=True)
        shutil.copy(img,os.path.join("sorting",str(key)))
except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)