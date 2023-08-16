import os
import pandas as pd

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models # 이미지
from torchvision import transforms
from PIL import Image

from transformers import BertTokenizer, BertModel # 텍스트

from tqdm.auto import tqdm
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

class Resnet_GPT2(nn.Module):
    def __init__(self, vocab_size):
        super(Resnet_GPT2, self).__init__()
        self.vocab_size = vocab_size

        self.resnet = models.resnet50(pretrained=True)
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.gpt2.resize_token_embeddings(vocab_size) # 추가한 [PAD] 토큰 반영

        combined_features_size = 1000 + self.gpt2.config.hidden_size # resnet 출력 차원 + gpt2 출력 차원
        self.classifier = nn.Linear(combined_features_size, vocab_size)

    def forward(self, images, question):
        image_features = self.resnet(images)
        image_features = image_features.view(image_features.size(0),-1)

        outputs = self.gpt2(question)
        output_features = outputs.last_hidden_state # [batch, sequence, hidden]

        image_features = image_features.unsqueeze(1).expand(-1, output_features.size(1),-1) # [batch, sequence, 1000]

        combined = torch.cat([image_features, output_features], dim=-1) # [batch, sequence, 1000+hidden]
        output = self.classifier(combined) # [batch, vocab_size]
        return output
    
class Resnet_BERT(nn.Module):
    def __init__(self, vocab_size):
        super(Resnet_BERT, self).__init__()
        self.vocab_size = vocab_size
        self.resnet = models.resnet50(pretrained=True)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert.resize_token_embeddings(vocab_size)

        combined_features_size = 1000 + self.bert.config.hidden_size # resnet 출력 차원 + bert 출력 차원
        self.classifier = nn.Linear(combined_features_size, vocab_size)

    def forward(self, images, question):
        image_features = self.resnet(images)
        image_features = image_features.view(image_features.size(0),-1)

        outputs = self.bert(question)
        output_features = outputs.last_hidden_state # [batch, sequence, hidden]

        image_features = image_features.unsqueeze(1).expand(-1, output_features.size(1),-1) # [batch, sequence, 1000]

        combined = torch.cat([image_features, output_features], dim=-1) # [batch, sequence, 1000+hidden]
        output = self.classifier(combined) # [batch, vocab_size]
        return output