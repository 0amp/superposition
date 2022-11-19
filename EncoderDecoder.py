import torch as t
import torch.nn as nn
from einops import repeat, rearrange

class LinearModel(nn.Module): 
    def __init__(self, input_dim, hidden_dim): 
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = self.linear1.weight

    def forward(self, x): 
        x = self.linear1(x)
        x = x @ self.linear2
        return x

class ReLUModel(nn.Module): 
    def __init__(self, input_dim, hidden_dim): 
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = self.linear1.weight

    def forward (self, x): 
        x = self.linear1(x)
        x = x @ self.linear2
        x = self.relu(x)
        return x

class EncoderDecoder(): 
    def __init__(self, Xs, I, hidden_dim, linear=True): 
        self.Xs = Xs
        self.I = repeat(I, 'd -> n d', n = Xs.shape[0])
        self.hidden_dim = hidden_dim
        self.linear = linear

        if self.linear: 
            self.model = LinearModel(Xs.shape[1], self.hidden_dim)
        else: 
            self.model = ReLUModel(Xs.shape[1], self.hidden_dim)

    def get_loss(self, detach=False): 
        Xs_prime = self.model(self.Xs)
        if not detach: 
            return (self.I * (Xs_prime - self.Xs)**2).sum()
        else: 
            return (self.I * (Xs_prime.detach() - self.Xs)**2).sum().item()

    def get_weights(self): 
        return self.model.linear1.weight.detach()

    def get_biases(self): 
        return rearrange(self.model.linear1.bias.detach(), 'd -> d 1')
    
    def forward(self, x): 
        return self.model(x)
    
    def train(self, epochs=100, lr=1e-3, verbose=False): 
        self.model.train()
        optimizer = t.optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(epochs): 
            optimizer.zero_grad()
            loss = self.get_loss()
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0 and verbose: 
                print(f"Epoch {epoch}: {loss}")