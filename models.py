import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchsummary import summary

"""************************************************Session8 Start************************************************"""

#Squeeze and Expand -- Xmas tree
class Session8_Model(nn.Module):    
    def __init__(self,dropout_value=0.1,number_of_groups= 2 normalization='batch'):
        # Initialize the Module class
        super(Session8_Model, self).__init__()

        self.dropout_value = dropout_value
        self.number_of_groups = number_of_groups

        # Convolutional Block-1
        self.conv_block1 = self.custom_conv_block(in_channels=3, out_channels=20, kernel_size=3, padding=1, normalization=normalization)
        self.conv_block2 = self.custom_conv_block(in_channels=20, out_channels=20, kernel_size=3, padding=1, normalization=normalization)

        # Transitional Block-1
        self.conv_point1 = self.custom_conv_block(in_channels=20, out_channels=16, kernel_size=1, padding=0, normalization=normalization)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block-2
        self.conv_block3 = self.custom_conv_block(in_channels=16, out_channels=26, kernel_size=3, padding=1, normalization=normalization)
        self.conv_block4 = self.custom_conv_block(in_channels=26, out_channels=26, kernel_size=3, padding=1, normalization=normalization)
        self.conv_block5 = self.custom_conv_block(in_channels=26, out_channels=26, kernel_size=3, padding=1, normalization=normalization)

        # Transitional Block-2
        self.conv_point2 = self.custom_conv_block(in_channels=26, out_channels=16, kernel_size=1, padding=0, normalization=normalization)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block-3
        self.conv_block7 = self.custom_conv_block(in_channels=16, out_channels=32, kernel_size=3, padding=1, normalization=normalization)
        self.conv_block8 = self.custom_conv_block(in_channels=32, out_channels=32, kernel_size=3, padding=1, normalization=normalization)
        self.conv_block9 = self.custom_conv_block(in_channels=32, out_channels=32, kernel_size=3, padding=1, normalization=normalization)

        # Global Average Pooling
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1))

        # Output Layer
        self.conv_block10 = self.custom_conv_block(in_channels=32, out_channels=10, kernel_size=1, padding=0, last_layer=True, normalization=normalization)

    def forward(self, x):
        # Convolutional Block-1
        x = self.conv_block1(x)
        x = x + self.conv_block2(x)

        # Transitional Block-
        x = self.conv_point1(x)
        x = self.pool1(x)

        # Convolutional Block-2
        x = self.conv_block3(x)
        x = x + self.conv_block4(x)
        x = x + self.conv_block5(x)

        # Transitional Block-2
        x = self.conv_point2(x)
        x = self.pool2(x)

        # Convolutional Block-3
        x = self.conv_block7(x)
        x = x + self.conv_block8(x)
        x = x + self.conv_block9(x)

        x = self.gap(x)
        x = self.conv_block10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)


    def get_normalization_layer(self,normalization,out_channels,number_of_groups = None):
        # Select normalization type
        if normalization == "layer":
            _norm_layer = nn.GroupNorm(1, out_channels)
        elif normalization == "group":
            if not number_of_groups:
                raise ValueError("Value of group is not defined")
            _norm_layer = nn.GroupNorm(number_of_groups, out_channels)
        else:
            _norm_layer = nn.BatchNorm2d(out_channels)
        
        return _norm_layer

    def custom_conv_block(self, in_channels: int,
                            out_channels: int,
                            kernel_size: int = 3,
                            padding: int = 0,
                            stride: int = 1,
                            normalization: str = "batch",
                            last_layer: bool = False):
        """
        Method to return a standard convolution block
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Size of the kernel used in the layer
        :param padding: Padding used in the layer
        :param stride: Stride used for convolution
        :param normalization: Type of normalization technique used
        :param last_layer: Flag to indicate if the layer is last convolutional layer of the network
        """
        _norm_layer = self.get_normalization_layer(self,normalization,out_channels,self.number_of_groups)

        # For last layer only return the convolution output
        if last_layer:
            return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size, bias=False, padding=padding))
        
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size, bias=False, padding=padding),
            nn.ReLU(),
            _norm_layer,
            nn.Dropout(self.dropout_value)
        )



"""************************************************Session8 End************************************************"""


"""************************************************Session7 Start************************************************"""
class Session7_Model10(nn.Module):
    def __init__(self,dropout_value):
        super(Session7_Model10, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 28 > 26 | 3

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 26 > 24 | 5

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 24 > 22 | 7

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10)
        ) # 22 > 20 | 9

        self.one_cross_one_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10)
        ) # 20 > 20 | 9

        self.trans1 = nn.Sequential(
          nn.MaxPool2d(2,2)
        )   # 20 > 10 | 10

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14)
        ) # 10 > 8 | 14

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14)
        )  # 8 > 6 | 18

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=13, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(13)
        )   #  6 > 4 | 22

        self.gap = nn.AdaptiveAvgPool2d(1)                             # 4 > 1 | 28

        self.one_cross_one_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=13, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10)
        )   # 1 > 1 | 28


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.one_cross_one_conv1(x)
        x = self.trans1(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.gap(x)
        x = self.one_cross_one_conv2(x)
        
        x = x.view(-1, 10)    
        return F.log_softmax(x, dim=-1)

class Session7_Model9(nn.Module):
    def __init__(self,dropout_value):
        super(Session7_Model9, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 28 > 26 | 3

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 26 > 24 | 5

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 24 > 22 | 7

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10)
        ) # 22 > 20 | 9

        self.one_cross_one_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10)
        ) # 20 > 20 | 9

        self.trans1 = nn.Sequential(
          nn.MaxPool2d(2,2)
        )   # 20 > 10 | 10

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14)
        ) # 10 > 8 | 14

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14)
        )  # 8 > 6 | 18

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=13, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(13)
        )   #  6 > 4 | 22

        self.gap = nn.AdaptiveAvgPool2d(1)                             # 4 > 1 | 28

        self.one_cross_one_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=13, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10)
        )   # 1 > 1 | 28



    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.one_cross_one_conv1(x)
        x = self.trans1(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.gap(x)
        x = self.one_cross_one_conv2(x)        
        x = x.view(-1, 10)    
        return F.log_softmax(x, dim=-1)

class Session7_Model8(nn.Module):
    def __init__(self,dropout_value):
        super(Session7_Model8, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 28 > 26 | 3

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 26 > 24 | 5

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 24 > 22 | 7

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10)
        ) # 22 > 20 | 9

        self.one_cross_one_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10)
        ) # 20 > 20 | 9

        self.trans1 = nn.Sequential(
          nn.MaxPool2d(2,2)
        )   # 20 > 10 | 10

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14)
        ) # 10 > 8 | 14

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14)
        )  # 8 > 6 | 18

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=13, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(13)
        )   #  6 > 4 | 22

        self.gap = nn.AdaptiveAvgPool2d(1)                             # 4 > 1 | 28

        self.one_cross_one_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=13, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10)
        )   # 1 > 1 | 28



    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.one_cross_one_conv1(x)
        x = self.trans1(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.gap(x)
        x = self.one_cross_one_conv2(x)        
        x = x.view(-1, 10)    
        return F.log_softmax(x, dim=-1)

class Session7_Model7(nn.Module):
    def __init__(self,dropout_value):
        super(Session7_Model7, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 28 > 26 | 3

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 26 > 24 | 5

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 24 > 22 | 7

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10)
        ) # 22 > 20 | 9

        self.one_cross_one_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10)
        ) # 20 > 20 | 9

        self.trans1 = nn.Sequential(
          nn.MaxPool2d(2,2)
        )   # 20 > 10 | 10

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14)
        ) # 10 > 8 | 14

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14)
        )  # 8 > 6 | 18

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=13, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(13)
        )   #  6 > 4 | 22

        self.gap = nn.AdaptiveAvgPool2d(1)                             # 4 > 1 | 28

        self.one_cross_one_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=13, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10)
        )   # 1 > 1 | 28

        self.dropout = nn.Dropout(dropout_value)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.one_cross_one_conv1(x)
        x = self.trans1(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.dropout(x)
        x = self.conv7(x)
        x = self.gap(x)
        x = self.one_cross_one_conv2(x)
        
        x = x.view(-1, 10)    
        return F.log_softmax(x, dim=-1)

class Session7_Model6(nn.Module):
    def __init__(self,dropout_value):
        super(Session7_Model6, self).__init__()
        # Input Block 28  
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 26 

        # CONVOLUTION BLOCK 1  
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 24. 
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU()
        ) # output_size = 22 

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 11 
        

        # CONVOLUTION BLOCK 2 4  8
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 9 >>> 27
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU()
        ) # output_size = 7 >>> 25

        # OUTPUT BLOCK  remove oradd
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 7 >>> 25
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7) # 7>> 9... nn.AdaptiveAvgPool((1, 1))
        ) # output_size = 1
        

        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.dropout(x)
        x = self.convblock7(x)
        x = self.gap(x)
       
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
        
class Session7_Model5(nn.Module):
    def __init__(self,dropout_value):
        super(Session7_Model5, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 24
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU()
        ) # output_size = 22

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 11

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 9
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU()
        ) # output_size = 7

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 7
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(7, 7), padding=0, bias=False),
            # nn.BatchNorm2d(10), NEVER
            # nn.ReLU() NEVER!
        ) # output_size = 1

        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.dropout(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Session7_Model4(nn.Module):
    def __init__(self,dropout_value):
        super(Session7_Model4, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 24
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU()
        ) # output_size = 22

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 11

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 9
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU()
        ) # output_size = 7

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 7
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(7, 7), padding=0, bias=False),
            # nn.BatchNorm2d(10), NEVER
            # nn.ReLU() NEVER!
        ) # output_size = 1

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Session7_Model3(nn.Module):
    def __init__(self,dropout_value):
        super(Session7_Model3, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),

            nn.ReLU()
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),

            nn.ReLU()
        ) # output_size = 24
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),


            nn.ReLU()
        ) # output_size = 22

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 11

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),


            nn.ReLU()
        ) # output_size = 9
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 7

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 7
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(7, 7), padding=0, bias=False),
            # nn.BatchNorm2d(10), NEVER
            # nn.ReLU() NEVER!
        ) # output_size = 1

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Session7_Model2(nn.Module):
    def __init__(self,dropout_value):
        super(Session7_Model2, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 24
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 22

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 11

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 9
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 7

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 7
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(7, 7), padding=0, bias=False),
            # nn.ReLU() NEVER!
        ) # output_size = 1 7x7x10 | 7x7x10x10 | 1x1x10

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
        
class Session7_Model1(nn.Module):
    def __init__(self,dropout_value):
        super(Session7_Model1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) # 28>28 | 3
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 28 > 28 |  5
        self.pool1 = nn.MaxPool2d(2, 2) # 28 > 14 | 10
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # 14> 14 | 12
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1) #14 > 14 | 14
        self.pool2 = nn.MaxPool2d(2, 2) # 14 > 7 | 28
        self.conv5 = nn.Conv2d(256, 512, 3) # 7 > 5 | 30
        self.conv6 = nn.Conv2d(512, 1024, 3) # 5 > 3 | 32 | 3*3*1024 | 3x3x1024x10 |
        self.conv7 = nn.Conv2d(1024, 10, 3) # 3 > 1 | 34 | > 1x1x10

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        # x = F.relu(self.conv7(x))
        x = self.conv7(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)

"""************************************************Session7 End************************************************"""

"""************************************************Session6 Start************************************************"""

class Session6_Model(nn.Module):
    def __init__(self,dropout_value):
        super(Session6_Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 28 > 26 | 3

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 26 > 24 | 5

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 24 > 22 | 7

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10)
        ) # 22 > 20 | 9

        self.one_cross_one_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10)
        ) # 20 > 20 | 9

        self.trans1 = nn.Sequential(
          nn.MaxPool2d(2,2)
        )   # 20 > 10 | 10

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14)
        ) # 10 > 8 | 14

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14)
        )  # 8 > 6 | 18

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=13, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(13)
        )   #  6 > 4 | 22

        self.gap = nn.AdaptiveAvgPool2d(1)                             # 4 > 1 | 28

        self.one_cross_one_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=13, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10)
        )   # 1 > 1 | 28


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.one_cross_one_conv1(x)
        x = self.trans1(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.gap(x)
        x = self.one_cross_one_conv2(x)
        
        x = x.view(-1, 10)    
        return F.log_softmax(x, dim=-1)

class Session6_Model_old(nn.Module):
    def __init__(self,dropout_prob):
        super(Session6_Model_old, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 4, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 4, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=dropout_prob)            
        )
        self.trans1 = nn.Sequential(
          nn.MaxPool2d(2,2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=dropout_prob)
        )
        self.trans2 = nn.Sequential(
          nn.MaxPool2d(2,2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 24, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout2d(p=dropout_prob)
        )
        self.trans3 = nn.Sequential(
          nn.MaxPool2d(2,2)
        )
        self.conv4 = nn.Sequential(
             nn.Conv2d(24, 16, 3, padding=1, bias=False),
             nn.ReLU(),
             nn.BatchNorm2d(16),
             nn.Conv2d(16, 14, 3, padding=1, bias=False),
             )
        
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(14, 10) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.trans2(x)
        x = self.conv3(x)
        x = self.trans3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = self.flat(x)
        x = self.fc(x)
        x =  F.log_softmax(x ,dim=1)
        return x 

"""************************************************Session6 End************************************************"""

"""************************************************Session5 Start************************************************"""
class Session5_Model(nn.Module):
    def __init__(self):
        """
        Constructor method to initialize the neural network layers.
        """
        super(Session5_Model, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)

        # Fully Connected Layers
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):        
        x = F.relu(self.conv1(x), 2)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x), 2)
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
"""************************************************Session5 End************************************************"""

def get_summary(model, input_size) :   
    
    """
    Generate Model Summary Using torchsummary.

    This function provides a summary of the PyTorch model using the torchsummary library.
    It displays information such as the model architecture, number of parameters,
    and memory consumption.

    Parameters:
    - model (torch.nn.Module): PyTorch model for which the summary is to be generated.
    - input_size (tuple): Tuple representing the input size of the model, e.g., (channels, height, width).
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    network = model.to(device)
    return summary(network, input_size=input_size)

