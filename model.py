import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
# from liuling_sr_color.work1.Net import net_common as common
import net_common as common
from network_swinir import SwinIR as net
from hat_arch import HAT as HAT_Net


class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()
        # model
        self.device = device
        self.backbone = Backbone()

        self.rootup = nn.Upsample(scale_factor=2, mode='bicubic')
        self.rfb = common.RFB_Block2(ch_in=3, ch_out=8)  # 输入通道为3，输出通道为8
        self.root = nn.Sequential(
            common.RFB_Block2(ch_in=8, ch_out=32),  # 输入通道为8，输出通道为32
            common.RFB_Block2(ch_in=32, ch_out=64)  # 输入通道为32，输出通道为64
        )
        # self.rfb1 = common.RFB_Block2(ch_in=64, ch_out=128)
        self.upsample = common.ConvUpsampler(64, 64)  # 输入通道和输出通道位64

        self.conv_0_1 = common.default_conv(ch_in=64, ch_out=96, k_size=3, stride=2)
        self.conv_0_2 = common.default_conv(ch_in=96, ch_out=128, k_size=3, stride=2)
        self.conv_0_3 = common.default_conv(ch_in=128, ch_out=160, k_size=3, stride=2)
        self.conv_0_4 = common.default_conv(ch_in=160, ch_out=192, k_size=3, stride=2)

        self.conv_1_1 = common.default_conv(ch_in=64, ch_out=128, k_size=3, stride=1)
        self.conv_1_2 = common.default_conv(ch_in=128, ch_out=160, k_size=3, stride=2)
        self.conv_1_3 = common.default_conv(ch_in=160, ch_out=192, k_size=3, stride=2)

        self.conv_2_1 = common.default_conv(ch_in=256, ch_out=160, k_size=3, stride=1)
        self.conv_2_2 = common.default_conv(ch_in=160, ch_out=192, k_size=3, stride=2)

        self.conv_3_1 = common.default_conv(ch_in=512, ch_out=192, k_size=3, stride=1)

        self.conv_4_1 = common.default_conv(ch_in=160, ch_out=192, k_size=3, stride=2)

        self.att1 = SE_Block(ch_in=768)
        self.att2 = SE_Block(ch_in=640)

        self.conv_out_1 = common.default_conv(ch_in=192, ch_out=160, k_size=3, stride=1)
        self.conv_out_2 = common.default_conv(ch_in=160, ch_out=128, k_size=3, stride=1)

        self.cubicfilter = CubicFilter(num_in_channels=128, num_out_channels=128)

        self.inv3 = nn.Sequential(
            common.default_conv(ch_in=128, ch_out=96, k_size=1, stride=1),
            common.involution(channels=96, kernel_size=3, stride=1),
            common.ConvUpsampler(96, 96),
            common.default_conv(ch_in=96, ch_out=96, k_size=1, stride=1)
        )
        self.inv4 = nn.Sequential(
            common.default_conv(ch_in=96, ch_out=64, k_size=1, stride=1),
            common.involution(channels=64, kernel_size=3, stride=1),
            common.ConvUpsampler(64, 64),
            common.default_conv(ch_in=64, ch_out=64, k_size=1, stride=1)
        )

        self.conv_4_l = common.default_conv(ch_in=64, ch_out=32, k_size=3, stride=1, bias=True, group=False)
        self.conv_5_l = common.default_conv(ch_in=32, ch_out=16, k_size=3, stride=1, bias=True, group=False)
        self.conv_6_l = common.default_conv(ch_in=16, ch_out=1, k_size=1, stride=1, bias=True, group=False)

        self.conv_4_ab = common.default_conv(ch_in=64, ch_out=32, k_size=3, stride=1, bias=True, group=False)
        self.conv_5_ab = common.default_conv(ch_in=32, ch_out=16, k_size=3, stride=1, bias=True, group=False)
        self.conv_6_ab = common.default_conv(ch_in=16, ch_out=2, k_size=3, stride=1, bias=True, group=False)

        self.ps2 = nn.PixelShuffle(2)
        self.sf = ShuffleFeature1(3, 3)  # FCMB模块
        self.sf1 = ShuffleFeature1(64, 64)
        self.sf2 = ShuffleFeature1(8, 8)

    def forward(self, x):
        x = x.float()  # [6, 3, 128, 128]
        result1 = HAT_Net(upscale=2, in_chans=3, img_size=64, window_size=8, compress_ratio=3, squeeze_factor=30,
                          conv_scale=0.01, overlap_ratio=0.5,img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                          mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        '''
        result1 = net(upscale=2, in_chans=3, img_size=64, window_size=8,
                      img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                      mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        '''
        result1 = result1.to('cuda')
        x00 = result1(x)  # [6, 3, 256, 256]

        x01 = self.sf(x) + x  # FCMB [6, 3, 128, 128]

        x1_0, x1_1, x2_0, x3_0 = self.backbone(x01)
        # (64,128,128)VGG-1  (128,64,64)VGG-2  (256,32,32)VGG-3  (512,16,16)VGG-4
        x1_0_up = self.upsample(x1_0)  # 第二行的upsample[6, 64, 256, 256]

        # x001 = self.rootup(x00)  # Upsample 3, 256, 256
        x001 = self.sf(x00) + x00  # FCMB 3, 256, 256
        x001 = self.rfb(x001)  # RBF 8, 256, 256，输入通道为3，输出通道为8
        x001 = self.sf2(x001) + x001  # FCMB 8, 256, 256
        x001 = self.root(x001)  # RBFx2 [6, 64, 256, 256] 第二行的+操作

        x0_1 = self.rootup(x01)  # Upsample 3, 256, 256
        x0_1 = self.sf(x0_1) + x0_1  # FCMB 3, 256, 256
        x0_1 = self.rfb(x0_1)  # RBF 8, 256, 256，输入通道为3，输出通道为8
        x0_1 = self.sf2(x0_1) + x0_1  # FCMB 8, 256, 256
        x0_1 = self.root(x0_1) + x001 + x1_0_up  # RBFx2 64, 256, 256 第二行的+操作

        x0_2 = common.Mish(self.conv_0_1(x0_1))  # 96, 128, 128
        x0_3 = common.Mish(self.conv_0_2(x0_2))  # 128, 64, 64
        x0_4 = common.Mish(self.conv_0_3(x0_3))  # 160, 32, 32
        x0_5 = common.Mish(self.conv_0_4(x0_4))  # 192, 16, 16
        # VGG-1结束

        x1_2 = common.Mish(self.conv_1_2(x1_1)) + x0_4  # 160, 32, 32
        x1_3 = common.Mish(self.conv_1_3(x1_2)) + x0_5  # 192, 16, 16
        # VGG-2结束

        x2_1 = common.Mish(self.conv_2_1(x2_0)) + x1_2  # 160, 32, 32
        x2_2 = common.Mish(self.conv_2_2(x2_1)) + x1_3  # 192, 16, 16
        # VGG-3结束

        x3_1 = common.Mish(self.conv_3_1(x3_0)) + x2_2  # 192, 16, 16
        # VGG-4结束

        out_1 = torch.cat([x0_5, x1_3, x2_2, x3_1], dim=1)  # 将4个通道数量为192的分支的结果进行拼接
        out_1 = self.att1(out_1)
        out_1 = common.Mish(self.conv_out_1(self.ps2(out_1)))  # 160, 32, 32，拼接的结果先传入PixelShuffle在传入一个3x3conv

        out_2 = torch.cat([x0_4, x1_2, x2_1, out_1], dim=1)  # 将4个通道数量为160的分支的结果进行拼接
        out_2 = self.att2(out_2)
        out_2_1 = common.Mish(self.conv_out_2(self.ps2(out_2))) + x0_3 + x1_1
        # 128, 64, 64，拼接的结果先传入PixelShuffle在传入一个3x3conv，有加上两个别的分支的维度为(128x64x64)的特征图
        out_2 = self.cubicfilter.get_cubic_mask(out_2_1) * out_2_1  # 128, 64, 64  将结果传入立方滤波器

        x = common.Mish(self.inv3(out_2)) + x0_2  # 96, 128, 128，第一个SCB
        x = common.Mish(self.inv4(x)) + x0_1 + x1_0_up  # 64, 256, 256，第二个SCB

        l = common.Mish(self.conv_4_l(x))  # 32, 256, 256
        l = common.Mish(self.conv_5_l(l))  # 16, 256, 256
        l = common.Mish(self.conv_6_l(l))  # 1, 256, 256
        # 亮度l分支

        ab = common.Mish(self.conv_4_ab(x))  # 32, 256, 256
        ab = common.Mish(self.conv_5_ab(ab))  # 16, 256, 256
        ab = common.Mish(self.conv_6_ab(ab))  # 2, 256, 256
        # 颜色ab分支

        return l, ab

    def model_train(self, l_lr, l_hr, ab):
        out_l_hr, out_ab = self.forward(l_lr)
        loss_1 = self.l1_loss_fun(l_hr, out_l_hr)
        loss_2 = self.l2_loss_fun(ab, out_ab)
        loss = loss_1 + loss_2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_count.update(loss.item(), l_lr.shape[0])

    def save_pkl(self, save_path, save_name):

        save_content = {
            'model': self.state_dict(),
            'epoch': self.train_epoch
        }
        if save_name == 'best':
            save_content = {
                'model': self.state_dict(),
                'epoch': self.train_epoch,
                'psnr': self.best_psnr,
                'ssim': self.best_ssim
            }

        torch.save(save_content, save_path + save_name + '-' + str(self.train_epoch).zfill(4) + '.pkl')

    def load_pkl(self, pkl_path, mode='trained', load=True):
        '''
        :param pkl_path: pkl路径
        :param mode: trained or best
        :param load: bool,是否要加载模型（训练时不用加载最佳模型）
        :return:
        '''
        success_load = False
        if not os.path.exists(pkl_path):
            print('PKL:this pkl path dose not exit!')
        else:
            # 寻找pkl文件并加载
            pkl_list = sorted(os.listdir(pkl_path))

            if len(pkl_list) == 0:
                print('PKL:no ' + mode + ' model!')
            elif pkl_list[-1].find('.pkl') == -1:
                print('PKL:no ' + mode + ' model!')
            else:
                checkpoint = torch.load(pkl_path + pkl_list[-1])
                if load:
                    self.load_state_dict(checkpoint['model'])
                success_load = True
        if success_load:
            if mode == 'trained':
                self.train_epoch = checkpoint['epoch']
                print(mode.capitalize() + ' PKL(EPOCH:' + str(self.train_epoch - 1) + ') successfully loaded !')
            elif mode == 'best':
                self.best_epoch = checkpoint['epoch']
                self.best_psnr = checkpoint['psnr']
                self.best_ssim = checkpoint['ssim']
                print(mode.capitalize() + ' PKL(EPOCH:' + str(self.best_epoch - 1) + ') successfully loaded !')
        else:
            if mode == 'trained':
                self.train_epoch = 1
            elif mode == 'best':
                self.best_epoch = 1
                self.best_psnr = 0
                self.best_ssim = 0
        return self


# 3x3 conv
class Block(nn.Module):

    def __init__(self):
        """Initialisation for a lower-level DeepLPF conv block

        :returns: N/A
        :rtype: N/A

        """
        super(Block, self).__init__()

    def conv3x3(self, in_channels, out_channels, stride=1):
        """Represents a convolution of shape 3x3

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param stride: the convolution stride
        :returns: convolution function with the specified parameterisation
        :rtype: function

        """
        return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                         stride=stride, padding=1, bias=True)


# 3x3 conv + ReLu
class ConvBlock(Block, nn.Module):

    def __init__(self, num_in_channels, num_out_channels, stride=1):
        """Initialise function for the higher level convolution block

        :param in_channels:
        :param out_channels:
        :param stride:
        :param padding:
        :returns:
        :rtype:

        """
        super(Block, self).__init__()
        self.conv = self.conv3x3(num_in_channels, num_out_channels, stride=2)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        """ Forward function for the higher level convolution block

        :param x: Tensor representing the input BxCxWxH, where B is the batch size, C is the number of channels, W and H are the width and image height
        :returns: Tensor representing the output of the block
        :rtype: Tensor

        """
        img_out = self.lrelu(self.conv(x))
        return img_out


# 最大池化
class MaxPoolBlock(Block, nn.Module):

    def __init__(self):
        """Initialise function for the max pooling block

        :returns: N/A
        :rtype: N/A

        """
        super(Block, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """ Forward function for the max pooling block

        :param x: Tensor representing the input BxCxWxH, where B is the batch size, C is the number of channels, W and H are the width and image height
        :returns: Tensor representing the output of the block
        :rtype: Tensor

        """
        img_out = self.max_pool(x)
        return img_out


# 全局平均池化
class GlobalPoolingBlock(Block, nn.Module):

    def __init__(self, receptive_field):
        """Implementation of the global pooling block. Takes the average over a 2D receptive field.
        :param receptive_field:
        :returns: N/A
        :rtype: N/A

        """
        super(Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """Forward function for the high-level global pooling block

        :param x: Tensor of shape BxCxAxA
        :returns: Tensor of shape BxCx1x1, where B is the batch size
        :rtype: Tensor

        """
        out = self.avg_pool(x)
        return out


# 立方滤波器
class CubicFilter(nn.Module):

    def __init__(self, num_in_channels=64, num_out_channels=64, batch_size=1):
        """Initialisation function

        :param block: a block (layer) of the neural network
        :param num_layers:  number of neural network layers
        :returns: initialises parameters of the neural networ
        :rtype: N/A

        """
        super(CubicFilter, self).__init__()
        self.num_out_channels = num_out_channels
        self.cubic_layer1 = ConvBlock(num_in_channels, num_out_channels)  # 3x3 conv + ReLu
        self.cubic_layer2 = MaxPoolBlock()  # 最大池化
        self.cubic_layer3 = ConvBlock(num_out_channels, num_out_channels)  # 3x3 cov n+ ReLu
        self.cubic_layer4 = MaxPoolBlock()  # 最大池化
        self.cubic_layer5 = ConvBlock(num_out_channels, num_out_channels)  # 3x3 conv + ReLu
        self.cubic_layer6 = MaxPoolBlock()  # 最大池化
        self.cubic_layer7 = ConvBlock(num_out_channels, num_out_channels)  # 3x3 conv + ReLu
        self.cubic_layer8 = GlobalPoolingBlock(2)  # 全局平均池化
        self.fc_cubic = torch.nn.Linear(num_out_channels, num_out_channels * 10)  # cubic线性层定义全连接
        self.upsample = torch.nn.Upsample(size=(300, 300), mode='bilinear', align_corners=False)  # 上采样
        self.dropout = nn.Dropout(0.5)  # 防止过拟合

    def get_cubic_mask(self, feat):
        """Cubic filter definition

        :param feat: feature map
        :param img:  image
        :returns: cubic scaling map
        :rtype: Tensor

        """
        #######################################################
        ####################### Cubic #########################
        feat = feat.float()
        batchsize = feat.shape[0]
        feat_cubic = self.upsample(feat)

        x = self.cubic_layer1(feat)  # 3x3 conv + ReLu
        x = self.cubic_layer2(x)  # 最大池化
        x = self.cubic_layer3(x)  # 3x3 conv + ReLu
        x = self.cubic_layer4(x)  # 最大池化
        x = self.cubic_layer5(x)  # 3x3 conv + ReLu
        x = self.cubic_layer6(x)  # 最大池化
        x = self.cubic_layer7(x)  # 3x3 conv + ReLu
        x = self.cubic_layer8(x)  # 全局平均池化
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)

        R = self.fc_cubic(x)

        x = np.arange(1, feat.shape[2] + 1, 1) / feat.shape[2]
        # 从 1 到 feat.shape[2] 的等差数列，然后除以 feat.shape[2] 得到均匀分布在 0 到 1 之间的值。
        v = np.vander(x, increasing=False)
        # 生成维数为 feat.shape[2] 的反范德蒙矩阵，increasing=False 表示生成的矩阵列的幂次按照递减顺序排列
        x_axis = torch.from_numpy(v)
        y_axis = torch.from_numpy(np.transpose(v))

        a = torch.zeros((1, 10, feat.shape[2], feat.shape[3])).cuda()
        # 创建了一个形状为 (1, 10, feat.shape[2], feat.shape[3]) 的全零张量 a
        a[0][0] = x_axis ** 3
        a[0][1] = (x_axis ** 2) * y_axis
        a[0][2] = (x_axis ** 2)
        a[0][3] = torch.ones_like(x_axis) + x_axis * (y_axis ** 2)
        a[0][4] = x_axis * y_axis
        a[0][5] = x_axis
        a[0][6] = y_axis ** 3
        a[0][7] = y_axis ** 2
        a[0][8] = y_axis
        a[0][9] = torch.ones_like(x_axis)
        a = a.repeat_interleave(self.num_out_channels, 0).unsqueeze(dim=0).repeat_interleave(batchsize, 0)
        # 对张量 a 进行了维度重复操作，首先沿着第 0 维重复 self.num_out_channels 次，然后在第 0 维上增加一个维度，最后再沿着第 0 维重复 batchsize 次
        b = R.view((batchsize, self.num_out_channels, 10, 1, 1))
        # step = 0
        # for i in range (self.num_out_channels):
        #     for j in range(10):
        #         b[:,i,j] = R[:, step]
        #         step += 1
        # b = b.unsqueeze(dim=3).unsqueeze(dim=4)
        # print(b.shape)
        cubic_mask = torch.sum(torch.mul(a, b), dim=2)
        # print(cubic_mask.shape)
        # print(torch.max(cubic_mask))
        # print(torch.min(cubic_mask))
        # exit(0)
        # print(R[0, 0])
        # print(R.shape)

        # for i in range(self.num_out_channels):
        #     cubic_mask[0, i, :, :] = self.find_mask_10(R=R, x_axis=x_axis, y_axis=y_axis, feat=feat, step=i)
        img_cubic = feat + cubic_mask
        # img_cubic = torch.clamp(feat + cubic_mask,0,1)
        return img_cubic


class Backbone(nn.Module):
    '''
    Backbone：vgg19的卷积层的0~26层
    '''

    def __init__(self):
        super(Backbone, self).__init__()
        # model
        self.model = models.vgg19(pretrained=True)
        '''
        torch.nn.Module.modules() 和 torch.nn.Module.children()
            modules(): 递归遍历所有子模块,包括子模块的子模块等
            children(): 只遍历所有直接子模块,不递归

        for child in model.children():
            print(child)
        '''
        self.smodel = list(self.model.children())[0]  # vgg19的卷积层
        self.model1 = self.smodel[:4]
        self.model2 = self.smodel[4:9]
        self.model3 = self.smodel[9:18]
        self.model4 = self.smodel[18:27]
        self.model5 = self.smodel[27:36]
        pass

    def forward(self, x):
        x = x.float()  # 3, 128, 128
        feature1 = self.model1(x)  # 64, 128, 128
        feature2 = self.model2(feature1)  # 128, 64, 64
        feature3 = self.model3(feature2)  # 256, 32, 32
        feature4 = self.model4(feature3)  # 512, 16, 16
        return feature1, feature2, feature3, feature4


# FCMB模块
class ShuffleFeature1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ShuffleFeature1, self).__init__()
        # model
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = common.default_conv(ch_in=1, ch_out=4, k_size=3)
        self.conv2 = common.default_conv(ch_in=4 * self.ch_in, ch_out=self.ch_out, k_size=3)  # 3x3 conv
        self.se = SE_Block1(self.ch_out, reduction=1)
        self.eca = ECA(channel=self.ch_in)
        self.inv = common.involution2(channels=4, kernel_size=3, stride=1)

    def forward(self, x):
        b, c, h, w = x.size()
        x1 = torch.split(x, 1, dim=1)  # (b, c, h, w)--->(b, 1, h, w)，u1....
        y = ()
        for i in x1:
            i = common.Mish(self.conv1(i))
            y1, y2 = torch.split(i, int(h / 2), dim=2)  # 沿着高度维度分割为两部分，分别赋值给 y1 和 y2,u11,u12......
            y1 = self.inv(y1, y2)  # DInv Block
            y = y.__add__((y1, y2))  # 以元组的形式存储y的结果
        for j in range(len(y)):
            if j % 2 == 0:
                if j == len(y) - 2:
                    y1 = torch.cat((y[j], y[1]), dim=2)
                else:
                    y1 = torch.cat((y[j], y[j + 3]), dim=2)
                if j == 0:
                    out = y1
                else:
                    out = torch.cat((out, y1), dim=1)
        out = common.Mish(self.conv2(out))  # 3x3 conv
        # out = self.se(out)
        # out = self.se(x, out)  # SE注意力机制
        out = self.eca(x, out)
        return out 


# SE注意力机制
class SE_Block1(nn.Module):
    def __init__(self, ch_in, reduction=8):
        super(SE_Block1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=True),  # 1x1x(r/c)
            nn.ReLU(inplace=True),  # 1x1x(r/c)
            nn.Linear(ch_in // reduction, ch_in, bias=True),  # 1x1xc
            nn.Sigmoid()
        )

    def forward(self, x, y):
        b, c, _, _ = y.size()
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# SE注意力机制
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=8):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=True),  # 1x1x(r/c)
            nn.ReLU(inplace=True),  # 1x1x(r/c)
            nn.Linear(ch_in // reduction, ch_in, bias=True),  # 1x1xc
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 全局自适应池化1x1xc
        y = self.fc(y).view(b, c, 1, 1)  # 全连接1x1xc
        return x * y.expand_as(x)  # y.expand_as(x)是HWC，权重和特征图相乘


class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        # feature descriptor on the global spatial information
        y = self.avg_pool(y)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)