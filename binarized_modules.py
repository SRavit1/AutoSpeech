#Adapted from SRavit1 3pxnet master branch
#3pxnet/3pxnet-training/binarized_modules_multi.py

import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
from datetime import datetime

import numpy as np
import utils_own
'''
def quantize(number,bitwidth):
    temp=1/bitwidth
    if number>0:
        for i in range(1,bitwidth):
            if number<=temp*i:
                return 2*i-1
        return 2*bitwidth-1
    else:
        for i in range(1,bitwidth):
            if number>=-temp*i:
                return -(2*i-1)
        return -(2*bitwidth-1)
'''


"""
def Binarize(tensor,quant_mode='det',bitwidth=1):
    if quant_mode == 'input':
        #return torch.round(tensor.mul_(45))
        return torch.clamp(tensor.mul_(45).div_(128),min=-0.99,max=0.99)
    if quant_mode=='multi':
        #tensor_clone = tensor.clone()
        #return tensor.sign()
        #temp = torch.floor(tensor.div_(2)).mul_(2).add_(1).mul_(tensor).div_(tensor)
        temp = torch.floor(tensor.mul_(2**bitwidth).div_(2)).mul_(2).add_(1).mul_(tensor).div_(tensor).div_(2**bitwidth)
        #temp = torch.floor(tensor_clone.mul_(2**bitwidth).div_(2)).mul_(2).add_(1).mul_(tensor_clone).div_(tensor_clone).div_(2**bitwidth)
        temp[temp!=temp]=0
        return temp
    if quant_mode=='det':
        return tensor.sign()
    if quant_mode=='bin':
        return (tensor>=0).type(type(tensor))*2-1
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

def Ternarize(tensor, mult = 0.7, mask = None, permute_list = None, pruned = False, align = False, pack = 32):
    if type(mask) == type(None):
        mask = torch.ones_like(tensor)
    
    # Fix permutation. Tensor needs to be permuted
    if not pruned:
        tensor_masked = utils_own.permute_from_list(tensor, permute_list)
        if len(tensor_masked.size())==4:
            tensor_masked = tensor_masked.permute(0,2,3,1)
       
        if not align:
            tensor_flat = torch.abs(tensor_masked.contiguous().view(-1)).contiguous()
            tensor_split = torch.split(tensor_flat, pack, dim=0)
            tensor_split = torch.stack(tensor_split, dim=0)
            tensor_sum = torch.sum(tensor_split, dim=1)
            tensor_size = tensor_sum.size(0)
            tensor_sorted, _ = torch.sort(tensor_sum)
            thres = tensor_sorted[int(mult*tensor_size)]
            tensor_flag = torch.ones_like(tensor_sum)
            tensor_flag[tensor_sum.ge(-thres) * tensor_sum.le(thres)] = 0
            tensor_flag = tensor_flag.repeat(pack).reshape(pack,-1).transpose(1,0).reshape_as(tensor_masked)
            
        else:
            tensor_flat = torch.abs(tensor_masked.reshape(tensor_masked.size(0),-1)).contiguous()
            tensor_split = torch.split(tensor_flat, pack, dim=1)
            tensor_split = torch.stack(tensor_split, dim=1)
            tensor_sum = torch.sum(tensor_split, dim=2)
            tensor_size = tensor_sum.size(1)
            tensor_sorted, _ = torch.sort(tensor_sum, dim=1)
            tensor_sorted = torch.flip(tensor_sorted, [1])
            multiplier = 32./pack
            index = int(torch.ceil((1-mult)*tensor_size/multiplier)*multiplier)
            thres = tensor_sorted[:, index-1].view(-1,1)
            tensor_flag = torch.zeros_like(tensor_sum)
            tensor_flag[tensor_sum.ge(thres)] = 1
            tensor_flag[tensor_sum.le(-thres)] = 1
            tensor_flag = tensor_flag.repeat(1,pack).reshape(tensor_flag.size(0),pack,-1).transpose(2,1).reshape_as(tensor_masked)

        if len(tensor_masked.size())==4:
            tensor_flag = tensor_flag.permute(0,3,1,2)            
        tensor_flag = utils_own.permute_from_list(tensor_flag, permute_list, transpose=True)
        tensor_bin = tensor.sign() * tensor_flag
            
    else:
        tensor_bin = tensor.sign() * mask
        
    return tensor_bin

#NOTE: output_bit unused parameter; remove in next revision
#Not changing now because it would break other code
class BinarizeLinear(nn.Linear):

    def __init__(self, input_bit=1, output_bit=1, weight_bit=1, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
        self.input_bit=input_bit
        self.output_bit=output_bit
        self.weight_bit = weight_bit

    def forward(self, input):
        #commented out below condition since multi binarization produces wrong result
        if True: #(input.size(1) != 768) and (input.size(1) != 3072): # 784->768
            input.data=Binarize(input.data,quant_mode='multi',bitwidth=self.input_bit)
        else:
            input.data = Binarize(input.data, quant_mode='det')
        weight_org_clone = self.weight_org.clone()
        weight_data=Binarize(self.weight_org, quant_mode="multi", bitwidth=self.weight_bit)
        self.weight.data=torch.clamp(weight_data,min=-0.99,max=0.99)
        self.weight_org = weight_org_clone #weight_org modified by Binarize function, we want it to stay the same
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out
    
#NOTE: output_bit unused parameter; remove in next revision
#Not changing now because it would break other code
class TernarizeLinear(nn.Linear):

    def __init__(self, thres, input_bit=1, output_bit=1, *kargs, **kwargs):
        try:
            pack = kwargs['pack']
        except:
            pack = 32
        else:
            del(kwargs['pack'])
        try:
            permute = kwargs['permute']
        except:
            permute = 1
        else:
            del(kwargs['permute'])
        try:
            self.align=kwargs['align']
        except:
            self.align=True
        else:
            del(kwargs['align'])
        super(TernarizeLinear, self).__init__(*kargs, **kwargs)
        self.input_bit = input_bit
        self.output_bit = output_bit
        permute = min(permute, self.weight.size(0))
        self.register_buffer('pack', torch.LongTensor([pack]))
        self.register_buffer('thres', torch.FloatTensor([thres]))
        self.register_buffer('mask', torch.ones_like(self.weight.data))
        self.register_buffer('permute_list', torch.LongTensor(np.tile(range(self.weight.size(1)), (permute,1))))
        self.register_buffer('weight_org', self.weight.data.clone())

    def forward(self, input, pruned=False):
        if (input.size(1) != 768) and (input.size(1) != 3072): # 784->768
            input.data=Binarize(input.data,quant_mode='multi',bitwidth=self.input_bit)
        else:
            input.data = Binarize(input.data, quant_mode='det')
        self.weight.data=Ternarize(self.weight_org, self.thres, self.mask, self.permute_list, pruned, align=self.align, pack=self.pack.item())
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)
        return out

#NOTE: output_bit unused parameter; remove in next revision
#Not changing now because it would break other code
class BinarizeConv2d(nn.Conv2d):

    def __init__(self, input_bit=1, output_bit=1, weight_bit=1, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
        self.input_bit = input_bit
        self.output_bit = output_bit
        self.weight_bit = weight_bit
        self.binarize_input = True
        #self.exp=True

    def forward(self, input):
        " ""
        if input.size(1) != 3 and input.size(1) != 1:
            input.data = Binarize(input.data,quant_mode='multi',bitwidth=self.input_bit)
        else:
            input.data = Binarize(input.data, quant_mode='input', bitwidth=self.input_bit)
        " ""
        
        input.data = Binarize(input.data, quant_mode="input", bitwidth=self.input_bit)
        if self.binarize_input:
          input.data = Binarize(input.data, quant_mode="multi", bitwidth=self.input_bit)
        #self.exp=True
        weight_org_clone = self.weight_org.clone()
        weight_data=Binarize(self.weight_org, quant_mode="multi", bitwidth=self.weight_bit)
        self.weight.data=torch.clamp(weight_data,min=-0.99,max=0.99)
        self.weight_org = weight_org_clone #weight_org modified by Binarize function, we want it to stay the same
        #if self.exp:
        #    now=datetime.now().time()
        #    temp=input.detach().numpy()
        #    with open ("./"+str(now.minute)+str(now.second)+str(now.microsecond)+".npy","wb") as f:
        #        np.save(f,temp)
        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)
        #if self.exp:
        #    now=datetime.now().time()
        #    temp=out.detach().numpy()
        #    with open("./" + str(now.minute) + str(now.second) + str(now.microsecond) + ".npy", "wb") as f:
        #        np.save(f, temp)

        return out

#NOTE: output_bit unused parameter; remove in next revision
#Not changing now because it would break other code
class TernarizeConv2d(nn.Conv2d):

    def __init__(self, thres, input_bit=1, output_bit=1, *kargs, **kwargs):
        try:
            pack = kwargs['pack']
        except:
            pack = 32
        else:
            del(kwargs['pack'])
        try:
            permute = kwargs['permute']
        except:
            permute = 1
        else:
            del(kwargs['permute'])
        try:
            self.align=kwargs['align']
        except:
            self.align=True
        else:
            del(kwargs['align'])
        self.input_bit = input_bit
        self.output_bit = output_bit
        self.binarize_input = True
        super(TernarizeConv2d, self).__init__(*kargs, **kwargs)
        permute = min(permute, self.weight.size(0))
        self.register_buffer('pack', torch.LongTensor([pack]))
        self.register_buffer('thres', torch.FloatTensor([thres]))
        self.register_buffer('mask', torch.ones_like(self.weight.data))
        self.register_buffer('permute_list', torch.LongTensor(np.tile(range(self.weight.size(1)), (permute,1))))
        self.register_buffer('weight_org', self.weight.data.clone())

    def forward(self, input, pruned=False):
        if self.binarize_input:
          input.data = Binarize(input.data, quant_mode="input", bitwidth=self.input_bit)
          input.data = Binarize(input.data, quant_mode="multi", bitwidth=self.input_bit)
        else:
          if input.size(1) != 3 and input.size(1) != 1:
              if self.input_bit==8:
                  input.data = torch.round(input.data.clamp_(-128,127))
              else:
                  input.data = Binarize(input.data,quant_mode='multi',bitwidth=self.input_bit)
          else:
              input.data = Binarize(input.data, quant_mode='input', bitwidth=self.input_bit)

        self.weight.data=Ternarize(self.weight_org, self.thres, self.mask, self.permute_list, pruned, align=self.align, pack=self.pack.item())
        #self.exp=True
        #if self.exp:
        #   now=datetime.now().time()
        #   temp=input.detach().numpy()
        #   with open ("./"+str(now.minute)+str(now.second)+str(now.microsecond)+".npy","wb") as f:
        #       np.save(f,temp)
        out = nn.functional.conv2d(input, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)
        #if self.exp:
        #    now=datetime.now().time()
        #    temp=out.detach().numpy()
        #    with open("./" + str(now.minute) + str(now.second) + str(now.microsecond) + ".npy", "wb") as f:
        #        np.save(f, temp)
        return out
"""

def Binarize(tensor,quant_mode='det',bitwidth=1):
    if quant_mode == 'input':
        #return torch.round(tensor.mul_(45))
        return torch.clamp(tensor.mul_(45).div_(128),min=-0.99,max=0.99)
    if quant_mode=='multi':
        temp = torch.floor(tensor.mul_(2**bitwidth).div_(2)).mul_(2).add_(1).mul_(tensor).div_(tensor).div_(2**bitwidth)
        temp[temp!=temp]=0
        return temp
    if quant_mode=='det':
        return tensor.sign()
    if quant_mode=='bin':
        return (tensor>=0).type(type(tensor))*2-1
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

#NOTE: output_bit unused parameter; remove in next revision
#Not changing now because it would break other code
class BinarizeConv2d(nn.Conv2d):

    def __init__(self, input_bit=1, output_bit=1, weight_bit=1, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
        self.input_bit = input_bit
        self.output_bit = output_bit
        self.weight_bit = weight_bit
        #self.exp=True

    def forward(self, input):
        input.data = Binarize(input.data, quant_mode="multi", bitwidth=self.input_bit)
        self.weight.data=torch.clamp(Binarize(self.weight_org.clone(), quant_mode="multi", bitwidth=self.weight_bit), min=-0.99, max=0.99)
        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        return out

def quantize(input, mask=None, quant=False, pruned=False, mult=0, bitwidth=8):
    if pruned:
        input = input * mask
    if quant:
        input = (input * np.power(2, bitwidth-1)).floor()/(np.power(2, bitwidth-1))
    if mult>0:
        input_flat = torch.abs(input.reshape(-1))
        input_size = input_flat.size(0)
        input_sorted, _ = torch.sort(input_flat)
        thres = input_sorted[int(mult*input_size)]
        input_flag = torch.ones_like(input_flat)
        input_flag[input_flat.ge(-thres) * input_flat.le(thres)] = 0
        mask = input_flag.reshape_as(input)
        input = input * mask
        return input, mask
    else:
        return input, torch.ones_like(input)
    
class QuantizeConv2d(nn.Conv2d):
    '''
    Quantized conv2d with mask for pruning
    '''
    def __init__(self, *kargs, bitwidth=8, weight_bitwidth=8, **kwargs):
        super(QuantizeConv2d, self).__init__(*kargs, **kwargs)
        self.bitwidth=bitwidth
        self.weight_bitwidth = weight_bitwidth
        #print("QuantizeConv2d initialized with bitwidth", self.bitwidth)
        self.thres = 0
        self.register_buffer('mask', torch.ones_like(self.weight.data))
        self.register_buffer('weight_org', self.weight.data.clone())
    
    def forward(self, input, quant=True, pruned=False):
        # If mult exists, overwrites pruning
        input.data, _ = quantize(input.data, quant=quant, bitwidth=self.bitwidth)
        self.weight.data, self.mask=quantize(self.weight_org, mask=self.mask, quant=quant, pruned=pruned, mult=self.thres, bitwidth=self.weight_bitwidth)
        out = F.conv2d(input, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
        return out
    
class QuantizeLinear(nn.Linear):
    '''
    Quantized Linear with mask for pruning
    '''
    def __init__(self, *kargs, bitwidth=8, weight_bitwidth=8, **kwargs):
        super(QuantizeLinear, self).__init__(*kargs, **kwargs)
        self.bitwidth=bitwidth
        self.weight_bitwidth = weight_bitwidth
        #print("QuantizeLinear initialized with bitwidth", self.bitwidth)
        self.thres = 0
        self.register_buffer('mask', torch.ones_like(self.weight.data))
        self.register_buffer('weight_org', self.weight.data.clone())
        
    def forward(self, input, quant=True, pruned=False, mult=None):
        input.data, _ = quantize(input.data, quant=quant, bitwidth=self.bitwidth)
        self.weight.data, self.mask=quantize(self.weight_org, mask=self.mask, quant=quant, pruned=pruned, mult=self.thres, bitwidth=self.weight_bitwidth)
        out = F.linear(input, self.weight)
        return out 
