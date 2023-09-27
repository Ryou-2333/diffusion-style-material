import torch
import torch.nn as nn

class MappingNet(nn.Module):
    def __init__(
        self,
        noise_size : int,
        context_size : int,
        context_embed_dim: int,
        depth : int,
        use_checkpoint=False,
        use_fp16=False,
    ):
        super().__init__()
        self.noise_size = noise_size
        self.context_size = context_size
        self.context_embed_dim = context_embed_dim
        self.hidden_size = noise_size
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.use_checkpoint = use_checkpoint
        self.depth = depth

        self.contex_embed = nn.Sequential(
            nn.Linear(context_size, context_embed_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(context_embed_dim, context_embed_dim)
        )

        self.emb_modeules = nn.ModuleList()
        self.module_list = nn.ModuleList()

        for _ in range(depth):
            self.module_list.append(nn.Sequential(
                nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1e-6),
                nn.LeakyReLU(0.2, True),
                nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            ))

            self.emb_modeules.append(nn.Linear(self.context_embed_dim, self.hidden_size))

        self.out = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.hidden_size, noise_size, bias=True),
        )

    def forward(self, x, context):
        c_emb = self.contex_embed(context)
        for i in range(self.depth):
            x = self.module_list[i](x)
            x += self.emb_modeules[i](c_emb)
        return self.out(x)