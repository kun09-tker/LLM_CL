import torch
import torch.nn as nn
import torch.nn.functional as F

class BertAdapter(nn.Module):
    def __init__(self, bert_hidden_size = 768, bert_adapter_size = 2000):
        super().__init__()
        self.fc1=torch.nn.Linear(bert_hidden_size, bert_adapter_size)
        self.fc2=torch.nn.Linear(bert_adapter_size, bert_hidden_size)
        self.activation = torch.nn.ReLU()
        print('BertAdapter')

    def forward(self,x):

        h=self.activation(self.fc1(x))
        h=self.activation(self.fc2(h))

        return x + h
        # return h

    def squash(self, input_tensor, dim=-1,epsilon=1e-16): # 0 will happen in our case, has to add an epsilon
        squared_norm = (input_tensor ** 2).sum(dim=dim, keepdim=True)
        squared_norm = squared_norm + epsilon
        scale = squared_norm / (1 + squared_norm)
        return scale * input_tensor / torch.sqrt(squared_norm)

class BertAdapterMask(BertAdapter):
    def __init__(self, ntasks, bert_hidden_size = 768, bert_adapter_size = 2000):
        super().__init__(bert_hidden_size=bert_hidden_size, bert_adapter_size=bert_adapter_size)
        self.efc1=torch.nn.Embedding(ntasks, bert_adapter_size)
        self.efc2=torch.nn.Embedding(ntasks, bert_hidden_size)
        self.gate=torch.nn.Sigmoid()
        print('BertAdapterMask')


    def forward(self,x,t,s):

        gfc1,gfc2=self.mask(t=t,s=s)
        h = self.get_feature(gfc1,gfc2,x)

        return x + h


    def get_feature(self,gfc1,gfc2,x):
        h=self.activation(self.fc1(x))
        h=h*gfc1.expand_as(h)

        h=self.activation(self.fc2(h))
        h=h*gfc2.expand_as(h)

        return h
    def mask(self,t,s=1):

       efc1 = self.efc1(torch.LongTensor([t]).cuda())
       efc2 = self.efc2(torch.LongTensor([t]).cuda())

       gfc1=self.gate(s*efc1)
       gfc2=self.gate(s*efc2)

       return [gfc1,gfc2]


    def my_softmax(self,input, dim=1):
        transposed_input = input.transpose(dim, len(input.size()) - 1)
        softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
        return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)