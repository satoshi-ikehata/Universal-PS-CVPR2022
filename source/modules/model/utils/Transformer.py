import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import kaiming_normal_, trunc_normal_
import math

class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False, attention_dropout = 0.1, dim_feedforward = 512):
        super(MultiHeadSelfAttentionBlock, self).__init__()
        self.dim_V = dim_out
        self.dim_Q = dim_in
        self.dim_K = dim_in
        self.num_heads = num_heads

        # Projection
        self.fc_q = nn.Linear(self.dim_Q, self.dim_V) # dimin -> dimhidden
        self.fc_k = nn.Linear(self.dim_K, self.dim_V) # dimin -> dimhidden
        self.fc_v = nn.Linear(self.dim_K, self.dim_V) # dimhidden -> dim

        if ln:
            self.ln0 = nn.LayerNorm(self.dim_Q)
            self.ln1 = nn.LayerNorm(self.dim_V)
        self.dropout_attention = nn.Dropout(attention_dropout)
        self.fc_o1 = nn.Linear(self.dim_V, dim_feedforward)
        self.fc_o2 = nn.Linear(dim_feedforward, self.dim_V)
        self.dropout1 = nn.Dropout(attention_dropout)
        self.dropout2 = nn.Dropout(attention_dropout)

    def forward(self, x, y):
        x = x if getattr(self, 'ln0', None) is None else self.ln0(x) # pre-normalization
        Q = self.fc_q(x) # input_dim -> embed dim       
        K, V = self.fc_k(y), self.fc_v(y) # input_dim -> embed dim
        dim_split = self.dim_V // self.num_heads # multi-head attention
        Q_ = torch.cat(Q.split(int(dim_split), 2), 0)
        K_ = torch.cat(K.split(int(dim_split), 2), 0)
        V_ = torch.cat(V.split(int(dim_split), 2), 0)
        A = self.dropout_attention(torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)) # Attention Dropout
        A =  A.bmm(V_) # A(Q, K, V) attention_output
        O = torch.cat((Q_ + A).split(Q.size(0), 0), 2)
        O_ = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        O = O + self.dropout2(self.fc_o2(self.dropout1(F.gelu(self.fc_o1(O_))))) 
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads=4, ln=False, attention_dropout = 0.1, dim_feedforward = 512):
        super(SAB, self).__init__()
        self.mab = MultiHeadSelfAttentionBlock(dim_in, dim_out, num_heads, ln=ln, attention_dropout = attention_dropout, dim_feedforward=dim_feedforward)
    def forward(self, X):
        return self.mab(X, X)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MultiHeadSelfAttentionBlock(dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class TransformerLayer(nn.Module):
    def __init__(self, dim_input, num_enc_sab = 3, num_outputs = 1, dim_hidden=384, dim_feedforward = 1024, num_heads=8, ln=False, attention_dropout=0.1):
        super(TransformerLayer, self).__init__()

        self.num_outputs = num_outputs
        self.dim_hidden = dim_hidden

        modules_enc = []
        modules_enc.append(SAB(dim_input, dim_hidden, num_heads, ln=ln, attention_dropout = attention_dropout, dim_feedforward=dim_feedforward))
        for k in range(num_enc_sab):
            modules_enc.append(SAB(dim_hidden, dim_hidden, num_heads, ln=ln, attention_dropout = attention_dropout, dim_feedforward=dim_feedforward))
        self.enc = nn.Sequential(*modules_enc)
        modules_dec = []
        modules_dec.append(PMA(dim_hidden, num_heads, num_outputs)) # after the PMA we should not put drop out
        self.dec = nn.Sequential(*modules_dec)
        print(f'Transformer (#Enc {num_enc_sab}, Dimhidden {dim_hidden}, DimFeedforward {dim_feedforward}, Norm {ln}, Dropout {attention_dropout})')
        

    def init_weights(self):
            for m in self.modules():            
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):# or isinstance(m, nn.Linear):
                    kaiming_normal_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.LayerNorm):
                    m.bias.data.zero_()
                    m.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        feat = x.view(-1, self.num_outputs * self.dim_hidden)
        return feat