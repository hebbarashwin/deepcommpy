import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN_Model(nn.Module):
    def __init__(self, config):
        super(RNN_Model, self).__init__()

        assert config['rnn_type'] in ['GRU', 'LSTM'], 'rnn_type must be GRU or LSTM'
        self.input_size = config['N']+1+int(config['onehot'])
        self.activation = config['activation']
        self.feature_size = config['rnn_feature_size']
        self.output_size = 1
        self.skip = config['use_skip']

        self.num_rnn_layers = config['rnn_depth']
        self.bidirectional = config['bidirectional']
        self.rnn = getattr(nn, config['rnn_type'])(self.input_size, self.feature_size, self.num_rnn_layers, bidirectional = self.bidirectional, batch_first = True)
        self.rnn_type = config['rnn_type']
        self.dropout = config['dropout']
        self.drop = nn.Dropout(config['dropout'])

        self.y_size = config['N']

        self.out_linear_depth = config['out_linear_depth']
        if config['use_layernorm']:
            self.layernorm = nn.LayerNorm(self.feature_size)
        else:
            self.layernorm = nn.Identity()

        if self.out_linear_depth == 1:
            self.linear = nn.Linear((int(self.bidirectional) + 1)*self.feature_size, self.output_size)
        else:
            layers = []
            layers.append(nn.Linear((int(self.bidirectional) + 1)*self.feature_size, self.y_hidden_size))
            for ii in range(1, self.out_linear_depth-1):
                layers.append(nn.SELU())
                layers.append(nn.Linear(self.y_hidden_size, self.y_hidden_size))
            layers.append(nn.SELU())
            layers.append(nn.Linear(self.y_hidden_size, self.output_size))
            self.linear = nn.Sequential(*layers)


    def act(self, inputs):
        if self.activation == 'tanh':
            return  F.tanh(inputs)
        elif self.activation == 'elu':
            return F.elu(inputs)
        elif self.activation == 'relu':
            return F.relu(inputs)
        elif self.activation == 'selu':
            return F.selu(inputs)
        elif self.activation == 'sigmoid':
            return F.sigmoid(inputs)
        elif self.activation == 'linear':
            return inputs
        else:
            return inputs

    def forward(self, input, hidden, Fy=None):

        out, hidden = self.rnn(input, hidden)
        out = self.drop(out)
        out = self.layernorm(out)

        if Fy is None:
            decoded = self.linear(out)
        else:
            decoded = self.linear(torch.cat([Fy, out], -1))
        decoded = decoded.view(-1, self.output_size)
        return decoded, hidden
    

class convNet(nn.Module):
    def __init__(self,config):
        super(convNet,self).__init__()
        self.hidden_dim = config.embed_dim
        self.input_len = config.max_len
        self.output_len = config.N
        bias = not config.dont_use_bias
        self.kernel = 7
        self.padding = int((self.kernel-1)/2)
        
        self.layers1 = nn.Sequential(
            nn.Conv1d(1,int(self.hidden_dim/2),self.kernel,padding=self.padding,bias=bias),
            nn.GELU(),
            nn.Conv1d(int(self.hidden_dim/2),int(self.hidden_dim/2),self.kernel,padding=2*self.padding,dilation=2,bias=bias),
            nn.GELU(),
            )
        self.layers2 = nn.Sequential(
            nn.Conv1d(int(self.hidden_dim/2),int(self.hidden_dim/2),self.kernel,padding=4*self.padding,dilation=4,bias=bias),
            nn.GELU(),
            nn.Conv1d(int(self.hidden_dim/2),int(self.hidden_dim/2),self.kernel,padding=self.padding,bias=bias),
            nn.GELU(),
            )
        self.layers3 = nn.Sequential(
            nn.Conv1d(int(self.hidden_dim/2),int(self.hidden_dim/2),self.kernel,padding=2*self.padding,dilation=2,bias=bias),
            nn.GELU(),
            nn.Conv1d(int(self.hidden_dim/2),int(self.hidden_dim/2),self.kernel,padding=4*self.padding,dilation=4,bias=bias),
            nn.GELU(),
            )
        self.layers4 = nn.Sequential(
            nn.Conv1d(int(self.hidden_dim/2),int(self.hidden_dim/2),self.kernel,padding=self.padding,bias=bias),
            nn.GELU(),
            nn.Conv1d(int(self.hidden_dim/2),int(self.hidden_dim/2),self.kernel,padding=2*self.padding,dilation=2,bias=bias),
            nn.GELU(),
            )
        self.layers5 = nn.Sequential(
            nn.Conv1d(int(self.hidden_dim/2),int(self.hidden_dim),self.kernel,padding=4*self.padding,dilation=4,bias=bias),
            nn.GELU(),
            nn.Conv1d(self.hidden_dim,self.hidden_dim,self.kernel,padding=self.padding,bias=bias),
            nn.GELU(),
            )
        self.layersFin = nn.Sequential(
            nn.Linear(self.hidden_dim*self.output_len , 4*self.output_len),
            nn.GELU(),
            nn.Linear(4*self.output_len , self.output_len),
            nn.GELU(),
            nn.Linear(self.output_len , self.output_len)
            )
            
        self.layer_norm = nn.LayerNorm(self.output_len, eps=1e-6)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self,noisy_enc):
        input1 = noisy_enc.unsqueeze(1)
        
        input2 = self.layers1(input1)
        
        residual2 = input2
        input3 = self.layers2(input2) + residual2

        residual3 = input3
        input4 =  self.layers3(input3)+ residual3
        
        residual4 = input4
        input5 =  self.layers4(input4) + residual4
        
        residual5 = input5
        input6 =  self.layers5(input5)
        
        
        output = self.layer_norm(self.dropout(self.layersFin(torch.flatten(input6,start_dim=1))))
        logits = output.squeeze()
        decoded_msg_bits = logits.sign()
        return decoded_msg_bits


