# 扩散模型超网架构
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.operations_gan import *
import models.genotypes as genotypes


#注意力机制
class ChannelAttention(nn.Module):
  def __init__(self, in_planes, ratio=16):
    super(ChannelAttention, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)  #全局平均池化
    self.max_pool = nn.AdaptiveMaxPool2d(1)  #全局最大池化
    # MLP
    self.fc1 = nn.Conv2d(in_planes, in_planes // 2, 1, bias=False)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Conv2d(in_planes // 2, in_planes, 1, bias=False)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
    max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
    out = avg_out + max_out
    return self.sigmoid(out)


class AttentionModule_up(nn.Module):
    def __init__(self, channel, ratio=16):
        super(AttentionModule_up, self).__init__()
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        attention_out = x * y.expand_as(x)

        op_attention = []
        op_channel = c // 3  # Number of channels per operation
        for i in range(3):
            temp = y[:, i * op_channel:op_channel * (i + 1), :, :]  # The attention weights of i-th operation
            op_i_atten = torch.sum(temp)  # Attention weights summation
            op_std = torch.std(temp)
            atten_std = op_i_atten.item() - op_std.item()
            op_attention.append(atten_std)

        return attention_out, op_attention


class AttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(AttentionModule, self).__init__()
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        attention_out = x * y.expand_as(x)

        op_attention = []
        op_channel = c // 6  # Number of channels per operation
        for i in range(6):
            temp = y[:, i * op_channel:op_channel * (i + 1), :, :]  # The attention weights of i-th operation
            op_i_atten = torch.sum(temp)  # Attention weights summation
            op_std = torch.std(temp)
            atten_std = op_i_atten.item() - op_std.item()
            op_attention.append(atten_std)

        return attention_out, op_attention


class MixedOp(nn.Module):

  def __init__(self, C_in, C_out, primitive_list, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.mp = nn.MaxPool2d(2, 2)
    self.k = 4
    self.ca = ChannelAttention(C_in)
    self.channel = C_in // self.k
    for primitive in primitive_list:
      op = OPS[primitive](C_in // self.k, C_out // self.k, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C_in, affine=False))
      self._ops.append(op)
    self.attention = AttentionModule(C_in * 6 // self.k, ratio=8)

  def forward(self, x):
      dim_2 = x.shape[1]
      num_list = self.ca(x)  # 计算注意力权值
      x = x * num_list  # 新输入
      slist = torch.sum(num_list, dim=0, keepdim=True)
      values, max_num_index = slist.topk(dim_2 // self.k, dim=1, largest=True, sorted=True)  # 获取权值最大的位置
      max_num_index = max_num_index.squeeze()
      num_dict = max_num_index
      xtemp = torch.index_select(x, 1, max_num_index)

      out = 0
      temp = []
      for op in self._ops:
          temp.append(op(xtemp))
      temp = torch.cat(temp[:], dim=1)  # Concatenate feature maps in channel dimension
      attention_out, op_attention = self.attention(temp)  # Calculate attention weights
      for i in range(6):  # Integrate all feature maps by element-wise addition
          out += attention_out[:, i * self.channel:self.channel * (i + 1):, :, :]
      # 将没有操作过的通道直接拼接
      if out.shape[2] == x.shape[2]:
          x[:, num_dict, :, :] = out[:, :, :, :]
      else:
          x = self.mp(x)
          x[:, num_dict, :, :] = out[:, :, :, :]
      return x, op_attention


class MixedOp_first(nn.Module):

  def __init__(self, C_in, C_out, stride, with_bn):
    super(MixedOp_first, self).__init__()
    self._ops = nn.ModuleList()
    if with_bn:
      PRIMITIVES = ['conv_1x1_bn', 'conv_3x3_bn', 'conv_5x5_bn']
    else:
      PRIMITIVES = ['conv_1x1', 'conv_3x3', 'conv_5x5']
    for primitive in PRIMITIVES:
      op = OPS[primitive](C_in, C_out, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))

class MixedOp_up(nn.Module):

  def __init__(self, C, primitives, stride=2):
    super(MixedOp_up, self).__init__()
    self._ops = nn.ModuleList()
    self.channel = C
    for primitive in primitives:
      op = OPS[primitive](C, C, stride, False)
      self._ops.append(op)
    self.attention = AttentionModule_up(C * 3, ratio=8)

  def forward(self, x):
      out = 0
      temp = []
      for up in self._ops:
          temp.append(up(x))
      temp = torch.cat(temp[:], dim=1)  # Concatenate feature maps in channel dimension
      attention_out, op_attention = self.attention(temp)  # Calculate attention weights
      for i in range(3):  # Integrate all feature maps by element-wise addition
          out += attention_out[:, i * self.channel:self.channel * (i + 1):, :, :]
      return out, op_attention


class MixedOp_down(nn.Module):

  def __init__(self, C, stride=2, skip=False):
    super(MixedOp_down, self).__init__()
    self._ops = nn.ModuleList()
    if skip:
      op = OPS['none'](C, stride, False)
      self._ops.append(op)
    for primitive in PRIMITIVES_DOWN:
      op = OPS[primitive](C, stride, False)
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))

class Cell_dis_Auto(nn.Module):
    def __init__(
            self,
            args,
            in_channels,
            out_channels,
            use_gumbel,
            hidden_channels=None,
            ksize=3,
            pad=1,
            activation=nn.ReLU(),
            downsample=False):
        super(Cell_dis_Auto, self).__init__()
        self._use_gumbel = use_gumbel
        self._t = args.t
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        
        PRIMITIVES = eval('genotypes.' + args.dis_normal_opr)
        self.c1 = MixedOp(in_channels, out_channels, PRIMITIVES, 1)
        self.c2 = MixedOp(out_channels, out_channels, PRIMITIVES, 1)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0)
            if args.d_spectral_norm:
                self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x, weights):
        h = x
        h = self.activation(h)
        if self._use_gumbel:
          a = calculate_gumbel_softmax(weights[0], self._t)
        else:
          a = F.softmax(weights[0], -1)
        h = self.c1(h, a)
        h = self.activation(h)
        if self._use_gumbel:
          a = calculate_gumbel_softmax(weights[1], self._t)
        else:
          a = F.softmax(weights[1], -1)
        h = self.c2(h, a)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x, weights):
        return self.residual(x, weights) + self.shortcut(x)

class Cell_dis_Auto_first(nn.Module):
    def __init__(
            self,
            args,
            in_channels,
            out_channels,
            use_gumbel,
            hidden_channels=None,
            ksize=3,
            pad=1,
            activation=nn.ReLU()):
        super(Cell_dis_Auto_first, self).__init__()
        self.activation = activation
        self.learnable_sc = (in_channels != out_channels)
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self._use_gumbel = use_gumbel
        self._t = args.t

        self.c1 = MixedOp_first(in_channels, out_channels, 1, args.dis_with_bn)
        PRIMITIVES = eval('genotypes.' + args.dis_normal_opr)  
        self.c2 = MixedOp(out_channels, out_channels, PRIMITIVES, 1)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0)
            if args.d_spectral_norm:
                self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x, weights_normal, weights_channels_raise):
        h = x
        h = self.activation(h)
        if self._use_gumbel:
          a = calculate_gumbel_softmax(weights_channels_raise[0], self._t)
        else:
          a = weights_channels_raise[0]
        h = self.c1(h, a)
        h = self.activation(h)
        if self._use_gumbel:
          a = calculate_gumbel_softmax(weights_normal[0], self._t)
        else:
          a = weights_normal[0]
        h = self.c2(h, a)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            return _downsample(x)
        else:
            return x

    def forward(self, x, weights_normal, weights_channels_raise):
        return self.residual(x, weights_normal, weights_channels_raise) + self.shortcut(x)

class Cell_gen_Auto(nn.Module):
  """
  Architecture of cell in Auto-GAN paper
  """
  def __init__(self, args, C_in, C_out, use_gumbel, prev=False, prev_prev=False):
    super(Cell_gen_Auto, self).__init__()

    self._use_gumbel = use_gumbel    

    self._mixed_ops_prev = nn.ModuleList()
    self._prev_prev = prev_prev
    self._t = args.t
    primitives_up = eval('genotypes.' + args.gen_up_opr)  # args.gen_up_opr is PRIMITIVES_UP
    if prev_prev:
      op = MixedOp_up(C_in, primitives_up, stride=4)
      self._mixed_ops_prev.append(op)
      # print('debug@: length of skip_up_op is {}'.format(len(op._ops)))
      op = MixedOp_up(C_in, primitives_up, stride=2)
      self._mixed_ops_prev.append(op)
      # print('debug@: length of skip_up_op is {}'.format(len(op._ops)))
    self._prev = prev
    if prev:
      op = MixedOp_up(C_in, primitives_up, stride=2)
      self._mixed_ops_prev.append(op)
      # print('debug@: length of skip_up_op is {}'.format(len(op._ops)))

    self._up_ops = nn.ModuleList()
    for i in range(2):
      op = MixedOp_up(C_in, primitives_up, stride=2)
      self._up_ops.append(op)

    self._normal_ops = nn.ModuleList()
    for i in range(3):
      primitives = eval('genotypes.' + args.gen_normal_opr)  # args.gen_normal_opr is PRIMITIVES_NORMAL_GEN_wo_skip_none_sep
      op = MixedOp(C_in, C_out, primitives, 1)
      self._normal_ops.append(op)
    
  def calculate_ops(self):
    len_up_ops = len(self._up_ops)
    len_norm_ops = len(self._normal_ops)
    len_skip_ops = len(self._mixed_ops_prev)

    return len_up_ops, len_norm_ops, len_skip_ops

  def forward(self, x, prev_ft=None, eval=False, gene=None, gene_skip=None):
    torch.autograd.set_detect_anomaly(True)
    # handle the skip features
    up_Attention = []
    op_Attention = []
    skip_ft = []
    if self._prev_prev:
      assert len(prev_ft) == 2
      for i in range(2):
        ft, up_attention = self._mixed_ops_prev[i](prev_ft[i])
        up_Attention.append(up_attention)
        skip_ft.append(ft)
    if self._prev:
      assert len(prev_ft) == 1
      for i in range(1):
        ft, up_attention = self._mixed_ops_prev[i](prev_ft[i])
        up_Attention.append(up_attention)
        skip_ft.append(ft)
    skip_ft = sum(ft for ft in skip_ft)

    # upsample the feature
    ft_up = []
    for i in range(len(self._up_ops)):
      ft, up_attention = self._up_ops[i](x)
      up_Attention.append(up_attention)
      ft_up.append(ft)
    ft_1 = ft_up[0]
    ft_3 = ft_up[1]

    # norm operation
    # calculate the output feature of node 2
    ft_2, op_attention = self._normal_ops[0](ft_1)
    op_Attention.append(op_attention)
    # norm operation
    # calculate the right input feature of node 4
    ft_4_right, op_attention = self._normal_ops[1](ft_2)
    op_Attention.append(op_attention)
    # norm operation (short cut)
    # calculate the left input feature of node 4
    ft_4_left, op_attention = self._normal_ops[2](ft_3)
    op_Attention.append(op_attention)
    ft_4 = ft_4_left + ft_4_right

    # add the skip feature from prev cell
    if self._prev or self._prev_prev:
      ft_4 = ft_4 + skip_ft

    return ft_1, ft_4, up_Attention, op_Attention

def calculate_gumbel_softmax(weights, t):
  noise = torch.rand(weights.shape).cuda()
  y = (weights - torch.log(-torch.log(noise + 1e-20))) / t
  out = F.softmax(y, dim=-1)
  return out