import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F

class conbr_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, stride, dilation):
        super(conbr_block, self).__init__()

        self.conv1 = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size, stride=stride, dilation = dilation, padding = 3, bias=True)
        self.bn = nn.BatchNorm1d(out_layer)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        out = self.relu(x)
        
        return out

class se_block(nn.Module):
    def __init__(self,in_layer, out_layer):
        super(se_block, self).__init__()
        
        self.conv1 = nn.Conv1d(in_layer, out_layer//8, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(out_layer//8, in_layer, kernel_size=1, padding=0)
        self.fc = nn.Linear(1,out_layer//8)
        self.fc2 = nn.Linear(out_layer//8,out_layer)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):

        x_se = nn.functional.adaptive_avg_pool1d(x,1)
        x_se = self.conv1(x_se)
        x_se = self.relu(x_se)
        x_se = self.conv2(x_se)
        x_se = self.sigmoid(x_se)
        
        x_out = torch.add(x, x_se)
        return x_out
        
class re_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, dilation):
        super(re_block, self).__init__()
        
        self.cbr1 = conbr_block(in_layer,out_layer, kernel_size, 1, dilation)
        self.cbr2 = conbr_block(out_layer,out_layer, kernel_size, 1, dilation)
        self.seblock = se_block(out_layer, out_layer)
    
    def forward(self,x):

        x_re = self.cbr1(x)
        x_re = self.cbr2(x_re)
        x_re = self.seblock(x_re)
        x_out = torch.add(x, x_re)
        return x_out  

class UNET_1D(nn.Module):
    def __init__(self, n_class, input_dim,layer_n,kernel_size,depth):
        super(UNET_1D, self).__init__()
        self.input_dim = input_dim
        self.n_class = n_class
        self.layer_n = layer_n
        self.kernel_size = kernel_size
        self.depth = depth
        
        self.AvgPool1D1 = nn.AvgPool1d(input_dim, stride=5)
        self.AvgPool1D2 = nn.AvgPool1d(input_dim, stride=25)
        self.AvgPool1D3 = nn.AvgPool1d(input_dim, stride=125)
        
        self.layer1 = self.down_layer(self.input_dim, self.layer_n, self.kernel_size,1, 2)
        self.layer2 = self.down_layer(self.layer_n, int(self.layer_n*2), self.kernel_size,5, 2)
        self.layer3 = self.down_layer(int(self.layer_n*2)+int(self.input_dim), int(self.layer_n*3), self.kernel_size,5, 2)
        self.layer4 = self.down_layer(int(self.layer_n*3)+int(self.input_dim), int(self.layer_n*4), self.kernel_size,5, 2)
        self.layer5 = self.down_layer(int(self.layer_n*4)+int(self.input_dim), int(self.layer_n*5), self.kernel_size,4, 2)

        self.cbr_up1 = conbr_block(int(self.layer_n*7), int(self.layer_n*3), self.kernel_size, 1, 1)
        self.cbr_up2 = conbr_block(int(self.layer_n*5), int(self.layer_n*2), self.kernel_size, 1, 1)
        self.cbr_up3 = conbr_block(int(self.layer_n*3), self.layer_n, self.kernel_size, 1, 1)
        self.upsample = nn.Upsample(scale_factor=5, mode='nearest')
        
        # decoder 1 classifier
        self.outcov = nn.Conv1d(self.layer_n, self.layer_n, kernel_size=self.kernel_size, stride=1,padding = 3)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.layer_n, self.n_class, bias=True)
        
        # decoder 2 classifier
        self.outcov2 = nn.Conv1d(self.layer_n, self.n_class, kernel_size=self.kernel_size, stride=1,padding = 3)
        self.fc2 = nn.Linear(self.n_class * 3500, self.n_class, bias=True)
        self.softmax = nn.Softmax(dim=1)
    
        
    def down_layer(self, input_layer, out_layer, kernel, stride, depth):
        block = []
        block.append(conbr_block(input_layer, out_layer, kernel, stride, 1))
        for i in range(depth):
            block.append(re_block(out_layer,out_layer,kernel,1))
        return nn.Sequential(*block)
            
    def forward(self, x):
        
        pool_x1 = self.AvgPool1D1(x)
        pool_x2 = self.AvgPool1D2(x)
        pool_x3 = self.AvgPool1D3(x)
        
        #############Encoder#####################
        
        out_0 = self.layer1(x)
        out_1 = self.layer2(out_0)
        
        x = torch.cat([out_1,pool_x1],1)
        out_2 = self.layer3(x)
        
        x = torch.cat([out_2,pool_x2],1)
        x = self.layer4(x)
        
        #############Decoder 1####################
        
        up1 = self.upsample(x)
        up1 = torch.cat([up1,out_2],1)
        up1 = self.cbr_up1(up1)
        
        up1 = self.upsample(up1)
        up1 = torch.cat([up1,out_1],1)
        up1 = self.cbr_up2(up1)
        
        up1 = self.upsample(up1)
        up1 = torch.cat([up1,out_0],1)
        up1 = self.cbr_up3(up1)
        
        out1 = self.outcov(up1)
        out1 = self.avgpool(out1)
        out1 = out1.view(out1.shape[0], -1)
        out1 = self.fc(out1)
        
        
        #############Decoder 2####################
        
        up2 = self.upsample(x)
        up2 = torch.cat([up2,out_2],1)
        up2 = self.cbr_up1(up2)
        
        up2 = self.upsample(up2)
        up2 = torch.cat([up2,out_1],1)
        up2 = self.cbr_up2(up2)
        
        up2 = self.upsample(up2)
        up2 = torch.cat([up2,out_0],1)
        up2 = self.cbr_up3(up2)
        
        out2 = self.outcov2(up2)
        out2 = torch.flatten(out2, start_dim=1)
        out2 = self.fc2(out2)
        out2 = self.softmax(out2)
       
        
        return out1 , out2
