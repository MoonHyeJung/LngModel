import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, num_layers=1):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define layers
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        # Get the embedding of the input
        embedded = self.embed(input)

        # Pass through RNN layer
        output, hidden = self.rnn(embedded, hidden)

        # Pass through the fully connected layer
        output = self.fc(output.reshape(output.size(0) * output.size(1), output.size(2)))

        return output, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden state
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, num_layers=1):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define layers
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        # Get the embedding of the input
        embedded = self.embed(input)

        # Pass through LSTM layer
        output, hidden = self.lstm(embedded, hidden)

        # Pass through the fully connected layer
        output = self.fc(output.reshape(output.size(0) * output.size(1), output.size(2)))

        return output, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden state
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))
