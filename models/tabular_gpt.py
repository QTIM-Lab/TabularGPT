import torch
import torch.nn as nn
from torch.nn import functional as F

from models.min_gpt import GPT, Block, NewGELU

class TabularGPT(GPT):
    def __init__(self, config, device, output_type="regression"):
        super().__init__(config)
        self.device = device
        self.output_type = output_type
        # Karpathy's had the output dim be vocab size, but ours is a flat dim
        # when predicting next time step, though this can be changed i.e.
        # if doing multi-class classification or something
        self.lm_head = nn.Linear(config.n_embd, config.out_dim, bias=False)
        self.transformer = nn.ModuleDict(dict(
            # wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.embd_pdrop),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))

        # Set transformer.wte below as modality fusion
        # self.transformer.wte = nn.ModuleDict(
        #     {
        #         f'var_n{i}': (nn.Embedding(config.embed_vars[str(i)],
        #                                    config.n_embd) if str(i) in config.embed_vars
        #                       else nn.Linear(1,
        #                                      config.n_embd,
        #                                      bias=True)) 
        #         for i in range(config.num_vars)
        #     }
        # )
        self.transformer.wte = nn.ModuleDict(
            {
                f'var_n{i}': (nn.Embedding(config.embed_vars[str(i)],
                                           config.n_embd) if str(i) in config.embed_vars
                              else nn.Sequential(nn.Linear(1,
                                             config.n_embd//2,
                                             bias=False), 
                                             nn.Linear(
                                                 config.n_embd//2, 
                                                 config.n_embd, 
                                                 bias=False))) 
                for i in range(config.num_vars)
            }
        )

        print('wte: ', self.transformer.wte)

        # save user specified embed variables
        self.embed_vars = config.embed_vars
        self.GELU = NewGELU()
        

    def freeze_projection_weights(self):
        for modality in self.modalities:
            for param in self.transformer.wte[modality].parameters():
                param.requires_grad = False

    def unfreeze_projection_weights(self):
        for modality in self.modalities:
            for param in self.transformer.wte[modality].parameters():
                param.requires_grad = True

    def forward(self, idx, targets=None):
        # device = idx.device
        b, t = idx.size()
        # b, t = next(iter(idx.values())).size()  # grab some dict value
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        # pos = torch.arange(0, t, dtype=torch.long, device=self.device).unsqueeze(0)  # shape (1, t)

        # forward the GPT model itself
        # tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        var_embeddings = []
        # for var_name in self.var_names:
        for i in range(t):
            if str(i) in self.embed_vars:
                var_embed = self.transformer.wte['var_n' + str(i)](idx[:,i].long())
            else:
                var_embed = self.transformer.wte['var_n' + str(i)](idx[:,i:(i+1)])
                var_embed = self.GELU(var_embed)
            var_embed = var_embed.unsqueeze(dim=1)
            var_embeddings.append(var_embed)
        # cat along "words" since each "word" has own embed
        # Shape is still (1, t, n_embd), same as pos_emb, but time step cat
        tok_emb = torch.cat(var_embeddings, dim=1)
        x = self.transformer.drop(tok_emb)
        # pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        # x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        # logits = self.lm_head(x)
        preds = self.lm_head(x)  # naming convention for time series vs tokens

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            # instead of cross entropy over classes, huber loss of time series
            # where we can also mask loss where target is neg inf
            # (for example if we only want to predict last time step
            # this is similar to cross-entropy's "ignore_index" and
            # Karpathy using -1 to ignore those masked tokens)
            # target_indices = ~torch.isinf(targets).any(dim=-1)
            target_indices = ~torch.isinf(targets)
            if self.output_type == "binaryclass":
                loss = F.binary_cross_entropy_with_logits(preds[target_indices].view(-1), targets[target_indices].view(-1))
                # loss = F.huber_loss(preds[target_indices], targets[target_indices])
            elif self.output_type == "multiclass":
                loss = F.cross_entropy(preds.view(-1, preds.size(-1)), targets.view(-1), ignore_index=-1)
            else:
                loss = F.huber_loss(preds[target_indices], targets[target_indices])
        return preds, loss