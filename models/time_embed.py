import math
import torch


class TimeEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        class NormalLinear(torch.nn.Linear):
            # From TGN code: From JODIE code
            def reset_parameters(self):
                stdv = 1.0 / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.embedding_layer = NormalLinear(1, self.out_channels)

    def forward(self, x, rel_t):
        embeddings = x * (1 + self.embedding_layer(rel_t.to(x.dtype).unsqueeze(1)))

        return embeddings