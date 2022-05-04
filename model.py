import torch.nn as nn
import torch,torchvision
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torch.utils import model_zoo



class ResNetEncoder(nn.Module):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        resnet34 = torchvision.models.resnet34()
        # self.in_channels = in_ch
        self.conv1 = resnet34.conv1
        self.bn1 = resnet34.bn1
        self.relu = resnet34.relu
        self.maxpool = resnet34.maxpool
        self.layer1 = resnet34.layer1
        self.layer2 = resnet34.layer2
        self.layer3 = resnet34.layer3
        self.layer4 = resnet34.layer4
        del resnet34.avgpool
        del resnet34.fc

    def forward(self,x):
        x1 = self.relu(self.bn1(self.conv1(x))) #64
        x2 = self.maxpool(self.layer1(x1)) #64
        x3 = self.layer2(x2) #128
        x4 = self.layer3(x3) #256
        x5 = self.layer4(x4) #512
        return [x1,x2,x3,x4,x5]

    def load_weights(self, state):
        state_dict = self.state_dict()
        for k, v in state.items():
            if 'fc' in k:
                continue
            state_dict.update({k: v})
        model = self.load_state_dict(state_dict)
        return model



class DecoderBlock(nn.Module):
    def __init__(self,in_ch,skip_ch,out_ch):
        super().__init__()
        self.conv1 = nn.Sequential(
              nn.Conv2d(in_channels=in_ch+skip_ch,out_channels=out_ch,kernel_size=(3,3),stride=1,padding=1,bias=False),
              nn.BatchNorm2d(num_features=out_ch),
              nn.ReLU(inplace=True)
                                  )
        self.conv2 = nn.Sequential(
              nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=(3,3),stride=1,padding=1,bias=False),
              nn.BatchNorm2d(num_features=out_ch),
              nn.ReLU(inplace=True)
                                  )
    
    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class SupervisionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=512,out_channels=128,kernel_size=(3,3),stride=1,padding=1,bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2))
        self.dense = nn.Linear(128*8*8,1)
    def forward(self,x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = x.view(x.size(0),-1)
        x = self.dense(x)
        return x

class ResUnet(nn.Module):
    def __init__(self,input_channels=3,classes=1,encoder_weights=True):
        super().__init__()
        self.encoder = ResNetEncoder()
        if encoder_weights:
          url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
          self.encoder.load_weights(model_zoo.load_url(url))

        self.classes = classes
        self.decoder1 = DecoderBlock(512,256,256)
        self.decoder2 = DecoderBlock(256,128,128)
        self.decoder3 = DecoderBlock(128,64,64)
        self.decoder4 = DecoderBlock(64,64,32)
        self.decoder5 = DecoderBlock(32,0,16)
        self.seg_head = nn.Conv2d(in_channels=16,out_channels=self.classes,kernel_size=(3,3),padding=3//2)
        self.supervision_block = SupervisionBlock()


    def forward(self, x):
        enc_outs = self.encoder(x)
        supervision_out = self.supervision_block(enc_outs[-1])

        x1 = self.decoder1(enc_outs[-1],enc_outs[-2])
        x2 = self.decoder2(x1,enc_outs[-3])
        x3 = self.decoder3(x2,enc_outs[-4])
        x4 = self.decoder4(x3,enc_outs[-5])
        x5 = self.decoder5(x4,skip=None)
        out = self.seg_head(x5)
        
        return out,supervision_out
