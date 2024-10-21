import torch
import torch.nn as nn
from torch.autograd import Variable
from MobileNetV2 import MobileNetV2, mobilenet_v2


class EventDetector(nn.Module):
    def __init__(self, pretrain, width_mult, lstm_layers, lstm_hidden, bidirectional=True, dropout=True):
        super(EventDetector, self).__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout

        # Load MobileNetV2 backbone
        net = mobilenet_v2(pretrained=False)
        state_dict_mobilenet = torch.load('/Users/davidromero/Documents/Capstone/Elaboration F24/ML/golfdb-master/mobilenet_v2.pth.tar', map_location=torch.device('cpu'))
        
        # Ensure we skip classifier layers that aren't needed for the model
        state_dict_mobilenet = {k: v for k, v in state_dict_mobilenet.items() if not k.startswith('classifier')}
        
        if pretrain:
            net.load_state_dict(state_dict_mobilenet, strict=False)

        # Use the convolutional layers from MobileNetV2
        self.cnn = nn.Sequential(*list(net.features))  # MobileNetV2 features

        # LSTM layer configuration
        self.rnn = nn.LSTM(int(1280 * width_mult if width_mult > 1.0 else 1280),
                           self.lstm_hidden, self.lstm_layers,
                           batch_first=True, bidirectional=bidirectional)
        
        # Fully connected layer based on whether LSTM is bidirectional
        if self.bidirectional:
            self.lin = nn.Linear(2 * self.lstm_hidden, 9)
        else:
            self.lin = nn.Linear(self.lstm_hidden, 9)
        
        # Dropout if enabled
        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def init_hidden(self, batch_size):
        # Initialize hidden states for LSTM
        if self.bidirectional:
            return (Variable(torch.zeros(2 * self.lstm_layers, batch_size, self.lstm_hidden).to('cpu'), requires_grad=True),
                    Variable(torch.zeros(2 * self.lstm_layers, batch_size, self.lstm_hidden).to('cpu'), requires_grad=True))
        else:
            return (Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).to('cpu'), requires_grad=True),
                    Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).to('cpu'), requires_grad=True))

    def forward(self, x, lengths=None):
        batch_size, timesteps, C, H, W = x.size()
        self.hidden = self.init_hidden(batch_size)

        # CNN forward pass
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        c_out = c_out.mean(3).mean(2)  # Global Average Pooling
        if self.dropout:
            c_out = self.drop(c_out)

        # LSTM forward pass
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, states = self.rnn(r_in, self.hidden)
        
        # Linear layer to get class predictions
        out = self.lin(r_out)
        out = out.view(batch_size * timesteps, 9)

        return out
