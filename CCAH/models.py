import math
import clip
import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pygat.layers import GraphAttentionLayer



class ImgNet(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(ImgNet, self).__init__()
        self.clip_image_encode, _ = clip.load("ViT-B/32", device="cuda:0")  # 512
        self.fc_encode = nn.Sequential(nn.Linear(512, 1024),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(1024, code_len)
                                        )
        self.decode = nn.Linear(code_len, txt_feat_len)
        self.alpha = 1.0

    def forward(self, x):
        with torch.no_grad():
            feat = self.clip_image_encode.encode_image(x)
            feat = feat.type(torch.float32)
        hid = self.fc_encode(feat)
        code = torch.tanh(self.alpha * hid)
        decoded_t = torch.sigmoid(self.decode(code))
        return feat, code, decoded_t

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class GATNet(nn.Module):
    def __init__(self, code_len, txt_feat_len, nhid=1024, image_size=1024, dropout=0.6, alpha=0.2, nheads=8):
        """Dense version of GAT."""
        super(GATNet, self).__init__()
        self.dropout = dropout
        self.decode = nn.Linear(code_len, image_size)
        self.alpha = 1.0

        self.attentions = [GraphAttentionLayer(txt_feat_len, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, code_len, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        code = torch.tanh(self.alpha * x)
        decoded_i = torch.sigmoid(self.decode(code))
        return code, decoded_i

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

