# flake8: noqa
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBNReLU(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding,
                      bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels,
                      dw_channels,
                      3,
                      stride,
                      1,
                      groups=dw_channels,
                      bias=False), nn.BatchNorm2d(dw_channels), nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(True))

    def forward(self, x):
        return self.conv(x)


class _DWConv(nn.Module):

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels,
                      out_channels,
                      3,
                      stride,
                      1,
                      groups=dw_channels,
                      bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))

    def forward(self, x):
        return self.conv(x)


class SE_Block(nn.Module):

    def __init__(self, in_channels, ratio=16):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio, bias=False),
            nn.ReLU(True),
            nn.Linear(in_channels // ratio, in_channels, bias=False),
            nn.Sigmoid())
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# 修改LinearBottleneck，添加SE模块
class LinearBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels

        # 主干网络
        self.block = nn.Sequential(
            # pw
            _ConvBNReLU(in_channels, in_channels * t, 1),
            # dw
            _DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels))

        # 添加SE模块
        self.se = SE_Block(out_channels, ratio=16)

    def forward(self, x):
        out = self.block(x)
        # SE模块处理
        out = self.se(out)
        # Shortcut连接
        if self.use_shortcut:
            out = x + out
        return out


class PyramidPooling(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


class Conv(nn.Module):

    def __init__(self,
                 nIn,
                 nOut,
                 kSize,
                 stride,
                 padding,
                 dilation=(1, 1),
                 groups=1,
                 bn_acti=False,
                 bias=False):
        super().__init__()
        self.bn_acti = bn_acti
        self.conv = nn.Conv2d(nIn,
                              nOut,
                              kernel_size=kSize,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)
        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)
        if self.bn_acti:
            output = self.bn_prelu(output)
        return output


class BNPReLU(nn.Module):

    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)
        return output


# 改进后的DASEModule，添加SE注意力机制
class DASEModule(nn.Module):

    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.conv1x1_in = Conv(nIn, nIn // 2, 1, 1, padding=0, bn_acti=False)

        # 第一分支
        self.conv3x1 = Conv(nIn // 2,
                            nIn // 2, (kSize, 1),
                            1,
                            padding=(1, 0),
                            bn_acti=True)
        self.conv1x3 = Conv(nIn // 2,
                            nIn // 2, (1, kSize),
                            1,
                            padding=(0, 1),
                            bn_acti=True)

        # 第二分支
        self.dconv3x1 = Conv(nIn // 2,
                             nIn // 2, (dkSize, 1),
                             1,
                             padding=(1, 0),
                             groups=nIn // 2,
                             bn_acti=True)
        self.dconv1x3 = Conv(nIn // 2,
                             nIn // 2, (1, dkSize),
                             1,
                             padding=(0, 1),
                             groups=nIn // 2,
                             bn_acti=True)
        self.se1 = SE_Block(nIn // 2, ratio=8)  # 添加SE模块，由于通道数减半，ratio也相应调整

        # 第三分支
        self.ddconv3x1 = Conv(nIn // 2,
                              nIn // 2, (dkSize, 1),
                              1,
                              padding=(1 * d, 0),
                              dilation=(d, 1),
                              groups=nIn // 2,
                              bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2,
                              nIn // 2, (1, dkSize),
                              1,
                              padding=(0, 1 * d),
                              dilation=(1, d),
                              groups=nIn // 2,
                              bn_acti=True)
        self.se2 = SE_Block(nIn // 2, ratio=8)  # 添加SE模块

        self.bn_relu_2 = BNPReLU(nIn // 2)
        self.conv1x1 = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)

        # 最终输出的SE模块
        self.se_out = SE_Block(nIn, ratio=16)  # 恢复原始通道数，使用标准ratio
        self.shuffle = ShuffleBlock(nIn // 2)

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv1x1_in(output)

        # 主路径
        main = self.conv3x1(output)
        main = self.conv1x3(main)

        # 深度可分离卷积分支1
        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)
        br1 = self.se1(br1)  # 应用SE注意力

        # 深度可分离卷积分支2（带膨胀）
        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)
        br2 = self.se2(br2)  # 应用SE注意力

        # 特征融合
        output = br1 + br2 + main
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)

        # 最终SE注意力和残差连接
        output = self.se_out(output)  # 应用最终的SE注意力
        output = self.shuffle(output + input)

        return output


class ShuffleBlock(nn.Module):

    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        #
        return x.view(N, g, int(C / g), H,
                      W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)


# 改进的REBNCONV，将SE模块放在残差连接之前
class REBNCONV(nn.Module):

    def __init__(self, in_ch=3, out_ch=3, dirate=1, use_se=True):
        super(REBNCONV, self).__init__()
        # 深度可分离卷积
        self.conv_dw = nn.Conv2d(in_ch,
                                 in_ch,
                                 kernel_size=3,
                                 padding=1 * dirate,
                                 dilation=1 * dirate,
                                 groups=in_ch,
                                 bias=False)
        # 逐点卷积
        self.conv_pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(True)
        # SE模块可选
        self.use_se = use_se
        if use_se:
            self.se = SE_Block(out_ch, ratio=8)  # ratio可以根据通道数调整

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv_dw(x)
        out = self.conv_pw(out)
        out = self.bn(out)
        out = self.relu(out)
        if self.use_se:
            out = self.se(out)
        return out


## 使用双线性插值进行上采样的基础函数 ##
def _upsample_like(src, tar):
    src = F.interpolate(src,
                        size=tar.shape[2:],
                        mode='bilinear',
                        align_corners=True)
    return src


# 改进后的RSU7模块
class RSU7(nn.Module):

    def __init__(self, in_ch=3, mid_ch=16, out_ch=56):
        super(RSU7, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1, use_se=False)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1, use_se=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1, use_se=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1, use_se=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1, use_se=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1, use_se=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1, use_se=True)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2, use_se=True)

        # 解码器部分
        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1, use_se=True)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1, use_se=True)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1, use_se=True)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1, use_se=True)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1, use_se=True)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1, use_se=True)

        # 残差连接前的SE模块
        self.se_out = SE_Block(out_ch, ratio=8)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        # 编码器路径
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)
        hx7 = self.rebnconv7(hx6)

        # 解码器路径
        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        # 在残差连接前应用SE注意力
        out = self.se_out(hx1d)

        # 残差连接
        return out + hxin


### RSU-6 模块，类似RSU7，调整膨胀率 ###
class RSU6(nn.Module):  # UNet06DRES(nn.Module):

    def __init__(self, in_ch=56, mid_ch=16, out_ch=56):
        super(RSU6, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

        # 残差连接前的SE模块
        self.se_out = SE_Block(out_ch, ratio=8)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        # 在残差连接前应用SE注意力
        out = self.se_out(hx1d)

        return out + hxin


### RSU-5 模块，类似RSU7，调整膨胀率 ###
class RSU5(nn.Module):  # UNet05DRES(nn.Module):

    def __init__(self, in_ch=56, mid_ch=16, out_ch=56):
        super(RSU5, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

        # 残差连接前的SE模块
        self.se_out = SE_Block(out_ch, ratio=8)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        # 在残差连接前应用SE注意力
        out = self.se_out(hx1d)

        return out + hxin


# 改进的下采样模块，结合DASEModule
class RSUDownsample(nn.Module):

    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64):
        super().__init__()

        # First downsampling with RSU7
        self.stage1 = RSU7(3, 16, dw_channels1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.DASE1 = DASEModule(dw_channels1, d=1)

        # Second downsampling with RSU6
        self.stage2 = RSU6(dw_channels1, 24, dw_channels2)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.DASE2 = DASEModule(dw_channels2, d=2)

        # Third downsampling with RSU5
        self.stage3 = RSU5(dw_channels2, 32, out_channels)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.DASE3 = DASEModule(out_channels, d=4)

    def forward(self, x):
        # First stage
        x = self.stage1(x)
        x = self.pool1(x)
        x = self.DASE1(x)

        # Second stage
        x = self.stage2(x)
        x = self.pool2(x)
        x = self.DASE2(x)

        # Third stage
        x = self.stage3(x)
        x = self.pool3(x)
        x = self.DASE3(x)

        return x


class GlobalFeatureExtractor(nn.Module):

    def __init__(self,
                 in_channels=64,
                 block_channels=(64, 96, 128),
                 out_channels=128,
                 t=6,
                 num_blocks=(3, 3, 3),
                 **kwargs):
        super(GlobalFeatureExtractor, self).__init__()

        # 第一个bottleneck序列和对应的DASE模块
        self.bottleneck1 = self._make_layer(LinearBottleneck,
                                            in_channels,
                                            block_channels[0],
                                            num_blocks[0],
                                            t,
                                            stride=2)
        self.DASE1 = nn.ModuleList(
            [DASEModule(block_channels[0], d=1) for _ in range(num_blocks[0])])

        # 第二个bottleneck序列和对应的DASE模块
        self.bottleneck2 = self._make_layer(LinearBottleneck,
                                            block_channels[0],
                                            block_channels[1],
                                            num_blocks[1],
                                            t,
                                            stride=2)
        self.DASE2 = nn.ModuleList(
            [DASEModule(block_channels[1], d=2) for _ in range(num_blocks[1])])

        # 第三个bottleneck序列和对应的DASE模块
        self.bottleneck3 = self._make_layer(LinearBottleneck,
                                            block_channels[1],
                                            block_channels[2],
                                            num_blocks[2],
                                            t,
                                            stride=1)
        self.DASE3 = nn.ModuleList(
            [DASEModule(block_channels[2], d=4) for _ in range(num_blocks[2])])

        # PPM模块保持不变
        self.ppm = PyramidPooling(block_channels[2], out_channels)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.ModuleList(layers)

    def forward(self, x):
        intermediate_features = []

        # 第一个bottleneck序列
        for i, bottleneck in enumerate(self.bottleneck1):
            x = bottleneck(x)
            x = self.DASE1[i](x)
            if i == 0:  # 存储第一个block的输出
                intermediate_features.append(x)

        # 第二个bottleneck序列
        for i, bottleneck in enumerate(self.bottleneck2):
            x = bottleneck(x)
            x = self.DASE2[i](x)
            if i == 0:  # 存储第二个block的输出
                intermediate_features.append(x)

        # 第三个bottleneck序列
        for i, bottleneck in enumerate(self.bottleneck3):
            x = bottleneck(x)
            x = self.DASE3[i](x)

        # PPM模块
        x = self.ppm(x)
        return x, intermediate_features


# Basic Residual Block
class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 添加SE模块
        self.se = SE_Block(out_channels, ratio=8)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 应用SE注意力
        out = self.se(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


# Bottleneck Block
class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()
        width = out_channels  # 瓶颈层通道数

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width,
                               width,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width,
                               out_channels * self.expansion,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        # 添加SE模块
        self.se = SE_Block(out_channels * self.expansion, ratio=16)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * self.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * self.expansion))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 应用SE注意力
        out = self.se(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class h_sigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):

    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):

    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class SSFEModule(nn.Module):

    def __init__(self, high_channels, low_channels):  # 修改初始化参数
        super(SSFEModule, self).__init__()
        self.high_channels = high_channels
        self.low_channels = low_channels

        # 添加通道对齐层
        self.channel_align = nn.Conv2d(low_channels,
                                       high_channels,
                                       kernel_size=1)

        # 定义MLP来计算权重
        self.mlp = nn.Sequential(
            nn.Conv2d(high_channels, high_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(high_channels // 2, high_channels, kernel_size=1))

        # 可学习的阈值
        self.threshold = nn.Parameter(torch.zeros(1))

    def forward(self, high_res, low_res):
        # 首先对低分辨率特征进行通道对齐
        aligned_low_res = self.channel_align(low_res)

        # 计算高分辨率和低分辨率特征图之间的差异
        diff_map = torch.mean((high_res - aligned_low_res)**2,
                              dim=1,
                              keepdim=True)

        # 二值化差异图
        binary_diff_map = (diff_map > self.threshold).float()

        # 计算特征图权重
        avg_pool = F.adaptive_avg_pool2d(high_res, (1, 1))
        max_pool = F.adaptive_max_pool2d(high_res, (1, 1))
        weights = torch.sigmoid(self.mlp(avg_pool + max_pool))

        # 重加权特征图
        reweighted_x = weights * high_res

        # 应用二值差异图进行特征增强
        enhanced_features = torch.mul(reweighted_x, binary_diff_map)

        return enhanced_features


# 修改高分辨率分支
class HighResolutionBranch(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 intermediate_channels=(64, 96)):  # 添加中间特征通道数参数
        super(HighResolutionBranch, self).__init__()

        # Basic Blocks with CA and SSFE
        self.basic_block1 = BasicBlock(in_channels, in_channels)
        self.ca1 = CoordAtt(in_channels, in_channels)
        # 第一个SSFE模块，对应第一个bottleneck block (64通道)
        self.SSFE1 = SSFEModule(high_channels=in_channels,
                                low_channels=intermediate_channels[0])

        self.basic_block2 = BasicBlock(in_channels, in_channels)
        self.ca2 = CoordAtt(in_channels, in_channels)
        # 第二个SSFE模块，对应第二个bottleneck block (96通道)
        self.SSFE2 = SSFEModule(high_channels=in_channels,
                                low_channels=intermediate_channels[1])

        # Bottleneck Block and CA
        self.bottleneck = BottleneckBlock(in_channels, out_channels // 4)
        self.ca3 = CoordAtt(out_channels, out_channels)

    def forward(self, x, low_res_feats):
        # 通过第一个Basic Block和CA
        x = self.basic_block1(x)
        x = self.ca1(x)
        # 第一个SSFE增强
        x = self.SSFE1(x, low_res_feats[0])

        # 通过第二个Basic Block和CA
        x = self.basic_block2(x)
        x = self.ca2(x)
        # 第二个SSFE增强
        x = self.SSFE2(x, low_res_feats[1])

        # 通过Bottleneck Block和CA
        x = self.bottleneck(x)
        x = self.ca3(x)

        return x


# 添加KAN模块所需的内层函数
class InnerFunction(nn.Module):
    """KANs内层函数Ψ实现"""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super(InnerFunction, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1), nn.BatchNorm2d(hidden_dim),
            nn.ReLU(), nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(),
            nn.Conv2d(hidden_dim, out_dim, 1))

    def forward(self, x):
        return self.net(x)


# 添加KAN模块所需的外层函数
class OuterFunction(nn.Module):
    """KANs外层函数Φ实现"""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super(OuterFunction, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(in_dim, hidden_dim, 1),
                                 nn.BatchNorm2d(hidden_dim), nn.ReLU(),
                                 nn.Conv2d(hidden_dim, out_dim, 1))

    def forward(self, x):
        return self.net(x)


# 添加改进的KANFusion用于两流特征融合
# 修改KANFusion以适应新的高分辨率分支
# 修改KANFusion以接收中间特征
# 修改KANFusion的初始化
# 双流融合网络：使用可学习的加权系数来动态调整不同特征流的融合比例
class DynKANFusion(nn.Module):

    def __init__(self,
                 higher_dim=64,
                 lower_dim=128,
                 inner_dim=256,
                 outer_dim=128):
        super(DynKANFusion, self).__init__()

        # 高分辨率分支
        self.high_res_branch = HighResolutionBranch(higher_dim,
                                                    lower_dim,
                                                    intermediate_channels=(64,
                                                                           96))

        # 内层函数
        self.inner_functions = nn.ModuleList(
            [InnerFunction(lower_dim, inner_dim, outer_dim) for _ in range(2)])

        # 外层函数
        self.outer_function = OuterFunction(outer_dim, outer_dim // 2,
                                            lower_dim)

        # 可学习的加权系数
        self.high_res_weight = nn.Parameter(torch.ones(1))  # 高分辨率流的权重
        self.low_res_weight = nn.Parameter(torch.ones(1))  # 低分辨率流的权重

    def forward(self, higher_res, lower_res, intermediate_features):
        h, w = higher_res.size()[2:]

        # 调整低分辨率特征大小
        up_lower_res = F.interpolate(lower_res,
                                     size=(h, w),
                                     mode='bilinear',
                                     align_corners=True)
        lower_res_feats = [
            F.interpolate(feat,
                          size=(h, w),
                          mode='bilinear',
                          align_corners=True) for feat in intermediate_features
        ]

        # 通过高分辨率分支
        higher_res = self.high_res_branch(higher_res, lower_res_feats)

        # 内层函数处理
        inner_out1 = self.inner_functions[0](higher_res)
        inner_out2 = self.inner_functions[1](up_lower_res)

        # 使用学习的权重对特征进行加权融合
        weighted_inner_out1 = inner_out1 * self.high_res_weight
        weighted_inner_out2 = inner_out2 * self.low_res_weight

        # 特征融合
        combined = weighted_inner_out1 + weighted_inner_out2

        return self.outer_function(combined)


class Deconv_BN_ACT(nn.Module):
    """增强版反卷积模块，用于特征图上采样和分辨率恢复"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=2,
                 use_se=True):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=kernel_size // 2,
                               output_padding=1,
                               bias=False), nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels))

        self.use_se = use_se
        if use_se:
            self.se = SE_Block(out_channels, ratio=8)

    def forward(self, x):
        x = self.deconv(x)
        if self.use_se:
            x = self.se(x)
        return x


class Classifer(nn.Module):
    """改进的分类器，结合反卷积上采样和DAP"""

    def __init__(self, dw_channels, num_classes, dap_k=3):
        super().__init__()
        self.dap_k = dap_k

        # 计算DAP所需的扩展类别数
        expanded_classes = num_classes * (dap_k**2)

        # 特征处理部分
        self.dsconv1 = _DSConv(dw_channels, dw_channels, 1)
        self.dsconv2 = _DSConv(dw_channels, dw_channels, 1)

        # 上采样路径
        self.deconv1 = Deconv_BN_ACT(dw_channels, dw_channels // 2)
        self.deconv2 = Deconv_BN_ACT(dw_channels // 2, dw_channels // 4)

        # 最终分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.1), nn.Conv2d(dw_channels // 4, expanded_classes, 1))

        # DAP模块
        self.dap = nn.Sequential(nn.PixelShuffle(dap_k),
                                 nn.AvgPool2d((dap_k, dap_k)))

    def forward(self, x):
        # 特征处理
        x = self.dsconv1(x)
        x = self.dsconv2(x)

        # 上采样
        x = self.deconv1(x)
        x = self.deconv2(x)

        # 分类
        x = self.classifier(x)

        # 应用DAP
        x = self.dap(x)

        return x


# self_net的其他部分保持不变
# 修改主网络以传递中间特征
class self_net(nn.Module):

    def __init__(self, n_classes=4, dap_k=3):
        super(self_net, self).__init__()
        self.n_classes = n_classes
        self.dap_k = dap_k

        # 下采样模块
        self.learning_to_downsample = RSUDownsample(32, 48, 64)

        # 特征提取器
        self.global_feature_extractor = GlobalFeatureExtractor(
            64, [64, 96, 128], 128, 6, [3, 3, 3])

        # KAN特征融合模块
        self.feature_fusion = DynKANFusion(higher_dim=64,
                                           lower_dim=128,
                                           inner_dim=256,
                                           outer_dim=128)

        # 分类器
        self.classifier = Classifer(128, n_classes, dap_k=dap_k)
        self.aux = False

    def forward(self, x):
        size = x.size()[2:]

        # 下采样路径
        higher_res_features = self.learning_to_downsample(x)

        # 全局特征提取，同时获取中间特征
        lower_res_features, intermediate_features = self.global_feature_extractor(
            higher_res_features)

        # 特征融合，传入中间特征
        fused_features = self.feature_fusion(higher_res_features,
                                             lower_res_features,
                                             intermediate_features)

        # 分类和DAP
        x = self.classifier(fused_features)

        # 上采样到原始尺寸
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x


# 创建一个输入张量
input_tensor = torch.randn(6, 3, 448, 448).cuda()  # 假设输入图像的尺寸为448x448，批量大小为6

# 初始化模型
model = self_net(n_classes=4).cuda()  # 假设有4个类别

# 将模型设置为评估模式
model.eval()

# 前向传播
with torch.no_grad():
    output = model(input_tensor)

# 打印输出张量的形状
print("Output shape:", output.shape)