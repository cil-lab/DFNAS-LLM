import torch
import torchvision
from torch import nn
from dataset import build_cifar10
import numpy as np
import torch.optim as optim
from torch import Tensor
from typing import Optional
from functools import partial
from torch.nn import functional as F


# network component 6 edge as graph, expands 
# use four node
# add encode for number of chenel


# change fc to one



type_dict = {0: 'zero', 1: 'direct', 2: 'nn.Conv2d_3*3', 3: 'nn.Conv2d_5*5',
             4: 'nn.AvgPool_3*3', 5: 'nn.MaxPool_3*3', 6: 'nn.Conv_1*1', 7: 'nn.Conv_3*3_dia'}
_A = 1 << np.arange(3)[::-1]
_C = 1 << np.arange(2)[::-1]


def covbin3(b: np.ndarray):
    return b.dot(_A).item()


def covbin2(b: np.ndarray):
    return b.dot(_C).item()


def decode_one(x: np.ndarray):
    num_block = 4
    
    # --- 1. Decode Types (0-40位) ---
    # 4个Block，每个Block 5个操作，每个操作2位
    # Index: 0 -> 40
    type_lists = []
    curr = 0
    for _ in range(num_block):
        block_types = []
        for _ in range(5):
            val = covbin2(x[curr : curr+2])
            block_types.append(val)
            curr += 2
        type_lists.append(block_types)

    # --- 2. Decode Links (40-64位) ---
    # 4个Block，每个Block 6条边，每条边1位
    # Index: 40 -> 64
    link_lists = []
    for _ in range(num_block):
        block_links = x[curr : curr+6]
        link_lists.append(block_links)
        curr += 6

    # --- 3. Decode Accents (64-72位) ---
    # Index: 64 -> 72
    accent_list = [covbin2(x[curr+i*2 : curr+(i+1)*2]) for i in range(4)]
    
    return type_lists, link_lists, accent_list


def gen_net10(x:np.ndarray):
    param=decode_one(x)
    return Net_10(*param)

class Net_10(nn.Module):
    def __init__(self, type_lists, link_lists, accent_list):
        super(Net_10, self).__init__()
        self.num_classes = 10
        self.num_block = 4
        self.type_lists = type_lists
        self.link_lists = link_lists
        self.accent_list = accent_list
        self.down_flag = [1,1,1,1]
        
        self.channel_nums=self.construct_channel()
        # print(self.channel_nums)
        self.main_layer = self.make_net()
        self.pre_layer = nn.Conv2d(3, self.channel_nums[0], 3, 1, 1)
        self.pre_mb = MBConv(self.channel_nums[0], self.channel_nums[1], expand_ratio=1, kernel=3)
        self.final_width = 32 >> (np.sum(self.down_flag))
        self.final_channels = self.channel_nums[-1]
        self.gap = nn.AdaptiveAvgPool2d(2)
        self.fc_width = self.final_channels * 4
        self.fc_width2 = 120
        self.fc1 = nn.Linear(self.fc_width, self.fc_width2)
        self.relu = nn.ReLU6(inplace=True)
        self.drop_out = nn.Dropout(0.3)
        self.fc2 = nn.Linear(self.fc_width2, self.num_classes)

    def construct_channel(self):
        first_channel=[16,20,24,32]
        accent=[1.0,1.2,1.4,1.6]
        ans=[32]
        ans.append(first_channel[self.accent_list[0]])
        for i in range(1,4):
            ans.append(int(ans[-1]*accent[self.accent_list[i]]))
        ans.append(ans[-1])
        return ans

    def make_net(self):
        self.main_net_list = []
        for i in range(self.num_block):
            in_channel = self.channel_nums[i+1]
            out_channel=self.channel_nums[i+2]
            block = Block(in_channel, type_list=self.type_lists[i], link_list=self.link_lists[i])
            self.main_net_list.append(block)
            self.main_net_list.append(nn.Conv2d(in_channel, out_channel, 3, 2))
            self.main_net_list.append(nn.BatchNorm2d(out_channel))

        return nn.Sequential(*self.main_net_list)

    def make_neck(self, neck_type):
        match neck_type:
            case 0:
                return nn.AvgPool2d(3, 1, 1)
            case 1:
                return nn.AvgPool2d(3, 2, 1)
            case 2:
                return nn.MaxPool2d(3, 1, 1)
            case 3:
                return nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        out = self.pre_layer(x)
        out = self.pre_mb(out)
        out = self.main_layer(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out=self.relu(out)
        out = self.drop_out(out)
        out = self.fc2(out)
        return out







class Net_100(nn.Module):
    def __init__(self, type_lists, link_lists, accent_list):
        super(Net_100, self).__init__()
        self.num_classes = 100
        self.num_block = 4
        self.type_lists = type_lists
        self.link_lists = link_lists
        self.accent_list = accent_list
        self.down_flag = [1,1,1,1]
        
        self.channel_nums=self.construct_channel()
        # print(self.channel_nums)
        self.main_layer = self.make_net()
        self.pre_layer = nn.Conv2d(3, self.channel_nums[0], 3, 1, 1)
        self.pre_mb = MBConv(self.channel_nums[0], self.channel_nums[1], expand_ratio=1, kernel=3)
        self.final_width = 32 >> (np.sum(self.down_flag))
        self.final_channels = self.channel_nums[-1]
        self.gap = nn.AdaptiveAvgPool2d(2)
        self.fc_width = self.final_channels * 4
        self.fc_width2 = 200
        self.fc1 = nn.Linear(self.fc_width, self.fc_width2)
        self.relu = nn.ReLU6(inplace=True)
        self.drop_out = nn.Dropout(0.4)
        self.fc2 = nn.Linear(self.fc_width2, self.num_classes)

    def construct_channel(self):
        first_channel=[16,20,24,32]
        accent=[1.0,1.2,1.4,1.6]
        ans=[32]
        ans.append(first_channel[self.accent_list[0]])
        for i in range(1,4):
            ans.append(int(ans[-1]*accent[self.accent_list[i]]))
        ans.append(ans[-1])
        return ans

    def make_net(self):
        self.main_net_list = []
        for i in range(self.num_block):
            in_channel = self.channel_nums[i+1]
            out_channel=self.channel_nums[i+2]
            block = Block(in_channel, type_list=self.type_lists[i], link_list=self.link_lists[i])
            self.main_net_list.append(block)
            self.main_net_list.append(nn.Conv2d(in_channel, out_channel, 3, 2))
            self.main_net_list.append(nn.BatchNorm2d(out_channel))

        return nn.Sequential(*self.main_net_list)

    def make_neck(self, neck_type):
        match neck_type:
            case 0:
                return nn.AvgPool2d(3, 1, 1)
            case 1:
                return nn.AvgPool2d(3, 2, 1)
            case 2:
                return nn.MaxPool2d(3, 1, 1)
            case 3:
                return nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        out = self.pre_layer(x)
        out = self.pre_mb(out)
        out = self.main_layer(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out=self.relu(out)
        out = self.drop_out(out)
        out = self.fc2(out)
        return out    


# class DepthwiseSeparableConv3(nn.Module):
#     def __init__(self, nin, kernels_per_layer, nout):
#         super(DepthwiseSeparableConv3, self).__init__()
#         self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin)
#         self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)

#     def forward(self, x):
#         out = self.depthwise(x)
#         out = self.pointwise(out)
#         return out
    
# class DepthwiseSeparableConv5(nn.Module):
#     def __init__(self, nin, kernels_per_layer, nout):
#         super(DepthwiseSeparableConv5, self).__init__()
#         self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=5, padding=2, groups=nin)
#         self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)

#     def forward(self, x):
#         out = self.depthwise(x)
#         out = self.pointwise(out)
#         return out

    

class MBConv(nn.Module):
    #  from deepseek-r1
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6, kernel=3):
        super(MBConv, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        hidden_dim = in_channels * expand_ratio
        self.use_res_connection = self.stride == 1 and in_channels == out_channels
        self.kernel=kernel
        if self.kernel==3:
            self.padding=1
        elif self.kernel==5:
            self.padding=2
        elif self.kernel==7:
            self.padding=3
            
        layers = []

        
        # 1. Expansion Phase（扩展阶段）
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # 2. Depthwise Convolution（深度可分离卷积）
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, self.kernel, stride=stride, padding=self.padding, 
                      groups=hidden_dim, bias=False),  # groups=hidden_dim 实现 Depthwise
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])
        
        # 3. Projection Phase（压缩阶段）
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_res_connection:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Block(nn.Module):
    def __init__(self, num_channel, type_list, link_list):
        super(Block, self).__init__()
        self.num_channel = num_channel
        self.link_list=link_list
        self.type_list = type_list
        # self.components=[]
        self.build()

    def build(self):
        self.layer0 = self.make_component(
            self.type_list[0], n_channel=self.num_channel)
        self.layer1 = self.make_component(
            self.type_list[1], n_channel=self.num_channel)
        self.layer2 = self.make_component(
            self.type_list[2], n_channel=self.num_channel)
        self.layer3 = self.make_component(
            self.type_list[3], n_channel=self.num_channel)
        self.layer4 = self.make_component(
            self.type_list[4], n_channel=self.num_channel)

    def make_component(self, type_num, n_channel):
        match type_num:
            case 0:
                return MBConv(n_channel,n_channel,kernel=3,expand_ratio=6)
            case 1:
                return MBConv(n_channel,n_channel,kernel=5,expand_ratio=6)
            case 2:
                return MBConv(n_channel,n_channel,kernel=3,expand_ratio=3)
            case 3:
                return MBConv(n_channel,n_channel,kernel=7,expand_ratio=6)



    def forward(self, x):
        line_1 = self.layer0(x)
        line_2 = self.layer1(line_1)
        line_3 = self.layer2(line_2 + self.link_list[0]*line_1)
        line_4 = self.layer3(line_3 + self.link_list[1]*line_1+self.link_list[3]*line_2)
        out = self.layer4(line_4 + self.link_list[2]*line_1+self.link_list[4]*line_2+self.link_list[5]*line_3)
        
        return out
    


class ResidualAdd(nn.Module):
    def __init__(self, block: nn.Module, shortcut: Optional[nn.Module] = None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut
        
    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        if self.shortcut:
            res = self.shortcut(res)
        x += res
        return x



class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        norm: nn.Module = nn.BatchNorm2d,
        act: nn.Module = nn.ReLU,
        **kwargs
    ):

        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            norm(out_features),
            act(),
        )


Conv1X1BnReLU = partial(ConvNormAct, kernel_size=1)
Conv3X3BnReLU = partial(ConvNormAct, kernel_size=3)



class SqueezeExcitation(nn.Module):
    def __init__(self,
                 input_c: int,   # block input channel
                 expand_c: int,  # block expand channel，DW卷积channel不变
                 squeeze_factor: int = 4):  # 控制第一个FC层节点个数，等于 input_c // squeeze_factor
        super(SqueezeExcitation, self).__init__()
        squeeze_c = input_c // squeeze_factor
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)    # 1x1卷积 代替全连接层，作用相同
        self.ac1 = nn.SiLU()  # alias Swish
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.ac2 = nn.Sigmoid()
        # print('------------')
        # print(input_c)
        # print(expand_c)
        # print(squeeze_c)
        # print('------------')

    def forward(self, x: Tensor) -> Tensor:
        # output_size=(1, 1)：对每个channel进行全局平均池化
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)     # 得到对应每个channel的对应程度
        return scale * x







# class MBConv3(nn.Sequential):
#     def __init__(self, in_features: int, out_features: int, expansion: int = 6):
#         residual = ResidualAdd if in_features == out_features else nn.Sequential
#         expanded_features = in_features * expansion
#         super().__init__(
#             nn.Sequential(
#                 residual(
#                     nn.Sequential(
#                         # narrow -> wide
#                         Conv1X1BnReLU(in_features, 
#                                     expanded_features,
#                                     act=nn.ReLU6
#                                     ),
#                         # wide -> wide
#                         # SqueezeExcitation(expanded_features  ,expanded_features),
#                         # Conv3X3BnReLU(expanded_features, 
#                         #             expanded_features, 
#                         #             groups=expanded_features,
#                         #             act=nn.ReLU6
#                         #             ),
#                         DepthwiseSeparableConv3(expanded_features,1,expanded_features),
                        
#                         Conv1X1BnReLU(expanded_features, out_features, act=nn.Identity),
#                     ),
#                 ),
#                 nn.ReLU(),
#             )
#         )


# class MBConv5(nn.Sequential):
#     def __init__(self, in_features: int, out_features: int, expansion: int = 6):
#         residual = ResidualAdd if in_features == out_features else nn.Sequential
#         expanded_features = in_features * expansion
#         super().__init__(
#             nn.Sequential(
#                 residual(
#                     nn.Sequential(
#                         # narrow -> wide
#                         Conv1X1BnReLU(in_features, 
#                                     expanded_features,
#                                     act=nn.ReLU6
#                                     ),
#                         # wide -> wide
#                         # se 
#                         # SqueezeExcitation(expanded_features,expanded_features),

#                         # Conv3X3BnReLU(expanded_features, 
#                         #             expanded_features, 
#                         #             groups=expanded_features,
#                         #             act=nn.ReLU6
#                         #             ),
#                         DepthwiseSeparableConv5(expanded_features,1,expanded_features),
                        
#                         Conv1X1BnReLU(expanded_features, out_features, act=nn.Identity),
#                     ),
#                 ),
#                 nn.ReLU(),
#             )
#         )







# class MBConv(nn.Module):
#     """
#     MBConv (Mobile Inverted Bottleneck) block as used in MobileNetV2.
#     Args:
#         in_channels (int):  输入通道数
#         out_channels (int): 输出通道数
#         expansion_ratio (int): 通道扩张倍数
#         stride (int):  步幅，决定空间下采样（默认为1或2）
#         kernel_size (int): 深度可分离卷积的卷积核大小，通常选3或5
#         use_res_connect (bool): 是否使用残差连接，当in_channels == out_channels且stride=1时一般为True
#     """
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  expand_ratio: int = 6,
#                  stride: int = 1,
#                  kernel: int = 3):
#         super(MBConv, self).__init__()

#         self.stride = stride
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         # 判断是否可以使用残差连接
#         self.use_res_connect = (self.stride == 1 and in_channels == out_channels)

#         # 中间层通道数
#         hidden_dim = in_channels * expand_ratio

#         # 1×1 升维卷积
#         self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1,
#                                      stride=1, padding=0, bias=False)
#         self.bn1 = nn.BatchNorm2d(hidden_dim)

#         # 3×3／5×5 的深度可分离卷积（Depthwise Conv）
#         self.depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel,
#                                         stride=stride, padding=kernel//2, groups=hidden_dim, bias=False)
#         self.bn2 = nn.BatchNorm2d(hidden_dim)

#         # 1×1 降维卷积
#         self.pointwise_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1,
#                                         stride=1, padding=0, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channels)

#         self.relu = nn.ReLU6(inplace=True)  # 在 MobileNetV2 开源实现中常使用 ReLU6

#     def forward(self, x):
#         out = self.expand_conv(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.depthwise_conv(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.pointwise_conv(out)
#         out = self.bn3(out)

#         if self.use_res_connect:
#             return x + out
#         else:
#             return out


if __name__=="__main__":
    # x=np.random.randint(0,2,24)
    # print(x)
    # print(decode_one(x))
    # x=np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1])
    # x=np.array([0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,1,1])
    x = np.random.randint(0, 2, 72)
    type_lists, link_lists,accent_list= decode_one(x)
    print(type_lists)
    print(link_lists)
    print(accent_list)
    # type_list=[[0, 0, 3, 1, 1, 3], [0, 3, 1, 1, 3, 0], [1, 1, 3, 0, 1, 3], [0, 3, 0, 0, 3, 0]]
    # neck_list=[1, 2, 1, 0]
    # flag_list=[1, 1, 1, 1]
    # net=Net_10(type_list,link_list,accent_list)
    
    
    # print(type_list)
    # print(neck_list)
    # print(flag_list)
