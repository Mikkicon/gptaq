from gptq import GPTQ
from quant import Quantizer
import torch
from scipy.stats import norm


class ActivationQuantizer(Quantizer):
  act_q_enabled = False

  def __init__(self):
    super(ActivationQuantizer, self).__init__()

  def quantize(self, x):
    return super().quantize(x)
  
  def configure(self, bits, **kvargs):
    self.act_q_enabled = bits < 16
    return super().configure(bits, **kvargs)


class GPTAQ(GPTQ):
  activation_quantizer: ActivationQuantizer
  out: torch.Tensor
  eig = False
  losses = []

  def __init__(self, layer, options = {}):
    self.eig = bool(options["eig"]) if "eig" in options else False
    super().__init__(layer)

  def add_batch(self, inp, out):
    if self.activation_quantizer.act_q_enabled:
      inp = self.activation_quantizer.quantize(inp)
      print("A Q input", end=" | ")
    return super().add_batch(inp, out)

  @staticmethod
  def cross_layer_equalization(layers):
      for i in range(len(layers) - 1):
          max_weight = torch.max(torch.abs(layers[i].weight))
          next_max_weight = torch.max(torch.abs(layers[i + 1].weight))
          scale = torch.sqrt(max_weight / next_max_weight)
          
          layers[i].weight.data /= scale
          layers[i + 1].weight.data *= scale
      return layers

  @staticmethod
  def CLE_2_layers(layer1, layer2):
      """s = 1/r2 * √(r1*r2)"""
      max_weight = torch.max(torch.abs(layer1.weight))
      next_max_weight = torch.max(torch.abs(layer2.weight))
      scale = torch.sqrt(max_weight / next_max_weight)
      layer1.weight.data /= scale
      layer2.weight.data *= scale

  @staticmethod
  def _bias_correction(beta, gamma):
    def gaussian_pdf(x):
      return torch.exp(-0.5 * x ** 2) / torch.sqrt(2 * torch.pi)
    def gaussian_cdf(x):
      return 0.5 * (1 + torch.erf(x / torch.sqrt(2)))
    beta = torch.tensor(beta, dtype=torch.float32)
    gamma = torch.tensor(gamma, dtype=torch.float32)
    z = -beta / gamma
    corrected_bias = gamma * gaussian_pdf(z) + beta * (1 - gaussian_cdf(z))
    return corrected_bias
  
  # TODO CLE + BC 
  def fdsa():
    if isinstance(module, torch.nn.BatchNorm2d.register_forward_hook):
        beta = module.bias.data
        gamma = module.weight.data
        corrected_bias = GPTAQ._bias_correction(beta, gamma)
        x = torch.functional.F.relu(module(x)) + corrected_bias
