from typing import Union
from gptq import GPTQ
from quant import Quantizer
import torch
from scipy.stats import norm


DEBUG=False

class GPTAQ(GPTQ):
  quantizer: Quantizer
  activation_quantizer: Quantizer
  out: torch.Tensor
  true_out: torch.Tensor
  eig = False
  reoptimize_W = False
  losses = []
  weights_quantized = False
  act_q_enabled = False
  biases_after_CLE = [] 
  biases = [] # corrected biases

  def __init__(self, layer, options = {}):
    super(GPTAQ, self).__init__(layer)
    self.eig = bool(options["eig"]) if "eig" in options else False
    self.reoptimize_W = bool(options["reoptimize"]) if "reoptimize" in options else False

  def configure(self, wbits, abits, **kvargs):
    self.act_q_enabled = abits < 16
    self.quantizer.configure(wbits, **kvargs)
    # self.activation_quantizer.configure(wbits, **kvargs)

  def after_forward(self, inp: torch.Tensor, out:torch.Tensor):
    quantize_activations = self.act_q_enabled and self.weights_quantized
    if DEBUG:
      print(inp.squeeze()[:3, :3])
      print(self.act_q_enabled, self.weights_quantized)
    
    if quantize_activations:
      if DEBUG:
        print(str(type(self.layer)), "weights quantized")
    
      if self.reoptimize_W:
        self.layer.weight = self.reoptimize_weights(inp, out)
    
      inp = self.activation_quantizer.quantize(inp)
      weights: torch.Tensor = self.layer.weight.data
      if DEBUG:
        print("weights shape", weights.shape, weights[:2, :3])
      
    #   if DEBUG:
    #     # max in inner-most tensors (row-wise)
    #     diffs = (weights.max(dim=-1)[0] - weights.min(dim=-1)[0]).abs()
    #     diffs, _ = diffs.sort(descending=True)
    #     print("20 channels ranges", diffs[:20])

      return
    elif not self.weights_quantized:
       self.true_out = out
      #  self.biases_after_CLE.append(self.layer.bias.data)
       self.weights_quantized = True
    return super().add_batch(inp, out)

  @staticmethod
  def MSE(x:torch.Tensor, q:torch.Tensor):
     return ((x - q) ** 2).mean()

  def reoptimize_weights(self, X: torch.Tensor, Y: torch.Tensor): 
      # Compute (X^T X)^-1 X^T Y
      if X.shape[0] != X.shape[1]:
         return self.layer.weight
      XTX_inv = torch.inverse(torch.matmul(X.T, X))
      XTY = torch.matmul(X.T, Y)
      W_opt = torch.matmul(XTX_inv, XTY)
      return W_opt

  @staticmethod
  def CLE_l1_l2(layer1: torch.nn.Module, layer2: torch.nn.Module):
      """s = 1/r2 * √(r1*r2)"""
      max_weight = torch.max(torch.abs(layer1.weight))
      next_max_weight = torch.max(torch.abs(layer2.weight))
      scale = torch.sqrt(max_weight / next_max_weight)
      # if DEBUG:
      #   print("before CLE",layer1.weight.data[0][:5])
      layer1.weight.data /= scale
      # if DEBUG:
      #   print("after CLE",layer1.weight.data[0][:5])
      layer2.weight.data *= scale


class RTNActQuantizer(Quantizer):
  act_q_enabled = False

  def __init__(self):
    super(RTNActQuantizer, self).__init__()

  def quantize(self, x, num_bits=8):
    qmin = 0.
    qmax = 2.**num_bits - 1.
    min_val, max_val = x.min(), x.max()
    
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - min_val / scale

    q_x = zero_point + x / scale
    q_x = torch.round(q_x)
    q_x = torch.clamp(q_x, qmin, qmax)

    dq_x = scale * (q_x - zero_point)
    return dq_x



class TokenWiseActQuantizer(Quantizer):
    def __init__(self, bitwidth=8):
        super(TokenWiseActQuantizer, self).__init__()
        self.bitwidth = bitwidth
        self.scale = None
        self.zero_point = None

    def forward(self, x):
      return self.quantize(x)
  
    def quantize(self, x):
        min_val, max_val = x.min(dim=-1, keepdim=True)[0], x.max(dim=-1,keepdim=True)[0]
        self.scale = (max_val - min_val) / (2 ** self.bitwidth - 1)
        self.zero_point = min_val
        quantized = torch.round((x - self.zero_point) / self.scale)
        return self.dequantize(quantized, self.scale, self.zero_point)

    def dequantize(self, quantized, scale, zero_point):
        return quantized * scale + zero_point



# OUTLIERS
      # x = torch.randn(1, 64, 32, 32)  # Example input tensor
      # x = inp
      # disassembled_x, outlier_indices = ChannelDisassembly(threshold=0.1)(x)
      # assembled_x = ChannelAssembly()(disassembled_x, original_channels=64)
      # print("Shapes: original -", x.shape, "  disassembled -", disassembled_x.shape, "  assembled -", assembled_x.shape)

# 	Disassembly step
class ChannelDisassembly(torch.nn.Module):
    def __init__(self, threshold=8):
        super(ChannelDisassembly, self).__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor):
        max_vals = torch.max(torch.abs(x.squeeze()))[0]
        outlier_indices = (max_vals > self.threshold).nonzero(as_tuple=True)[0]
        disassembled_channels = []
        
        # disassemble each outlier channel
        for idx in outlier_indices:
            T = torch.ceil(max_vals[idx] / self.threshold).int().item()
            sub_channels = x[:, idx, :, :].unsqueeze(1) / T
            disassembled_channels.append(sub_channels.repeat(1, T, 1, 1))
        
        disassembled_x = torch.cat(disassembled_channels, dim=1)
        return disassembled_x, outlier_indices


class ChannelAssembly(torch.nn.Module):
    def forward(self, x, original_channels):
        merged_channels = []
        channel_indices = list(range(x.size(1)))
        while len(channel_indices) > original_channels:
            # Find the two most similar channels
            min_distance = float('inf')
            min_pair = None
            for i in range(len(channel_indices)):
                for j in range(i + 1, len(channel_indices)):
                    c1 = channel_indices[i]
                    c2 = channel_indices[j]
                    dist = torch.norm(x[:, c1, :, :] - x[:, c2, :, :], p=2)
                    if dist < min_distance:
                        min_distance = dist
                        min_pair = (i, j)
            
            c1, c2 = min_pair
            new_channel=(x[:,channel_indices[c1],:,:] + x[:, channel_indices[c2],:,:])/2
            merged_channels.append(new_channel)
            
            # Remove merged channels and add the new one
            channel_indices.pop(max(c1, c2))
            channel_indices.pop(min(c1, c2))
            channel_indices.append(len(merged_channels) - 1)
        
        merged_x = torch.stack(merged_channels, dim=1)
        return merged_x



# class ActQuantWrapper(torch.nn.Module):
#     quantizer: Quantizer

#     def __init__(self, module, quantizer):
#         super(ActQuantWrapper, self).__init__()
#         self.module = module
#         shape = [1] * len(self.module.weight.shape)
#         if len(shape) == 4:
#             shape[1] = self.module.weight.shape[1]
#         if len(shape) == 3:
#             shape[2] = self.module.weight.shape[2]
#         if len(shape) == 2:
#             shape[1] = self.module.weight.shape[1]
#         self.quantizer = quantizer

#     def forward(self, x):
#         print("ActQuantWrapper.forward")
#         return self.module(self.quantizer.quantize(x))

# def add_actquant(module, name='', layers=[torch.nn.Conv2d, torch.nn.Linear], quantizer=None):
#     if isinstance(module, ActQuantWrapper):
#         return
#     for attr in dir(module):
#         tmp = getattr(module, attr)
#         if type(tmp) in layers:
#             setattr(module, attr, ActQuantWrapper(tmp, quantizer))
#         if type(tmp) == torch.nn.Sequential:
#             replaced = []
#             for i, child in enumerate(tmp.children()):
#                 if type(child) in layers:
#                     replaced.append(ActQuantWrapper(child, quantizer))
#                 else:
#                     replaced.append(child)
#             setattr(module, attr, torch.nn.Sequential(*replaced))
#         if type(tmp) == torch.nn.ModuleList:
#             replaced = []
#             for i, child in enumerate(tmp.children()):
#                 if type(child) in layers:
#                     replaced.append(ActQuantWrapper(child, quantizer))
#                 else:
#                     replaced.append(child)
#             setattr(module, attr, torch.nn.ModuleList(replaced))
#     for name1, child in module.named_children():
#         add_actquant(child, name + '.' + name1 if name != '' else name1, layers, quantizer)



