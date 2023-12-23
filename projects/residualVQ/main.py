import torch
from vector_quantize_pytorch import ResidualVQ

residual_vq = ResidualVQ(
    dim = 9,
    num_quantizers = 8,      # specify number of quantizers
    codebook_size = 1024,    # codebook size
)

x = torch.randn(1, 1024, 9)

quantized, indices, commit_loss = residual_vq(x)

import pdb; pdb.set_trace()
# (1, 1024, 256), (1, 1024, 8), (1, 8)
# (batch, seq, dim), (batch, seq, quantizer), (batch, quantizer)

# if you need all the codes across the quantization layers, just pass return_all_codes = True

quantized, indices, commit_loss, all_codes = residual_vq(x, return_all_codes = True)

# *_, (8, 1, 1024, 256)
# all_codes - (quantizer, batch, seq, dim)