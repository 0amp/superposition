import torch
import torch.nn as nn
from tqdm import tqdm
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
    def __init__(self, Xs, I, hidden_dim, linear=True, device='cpu'): 
        self.Xs = Xs.to(device)
        self.I = repeat(I, 'd -> n d', n = Xs.shape[0]).to(device)
        self.hidden_dim = hidden_dim
        self.linear = linear

        if self.linear: 
            self.model = LinearModel(Xs.shape[1], self.hidden_dim).to(device)
        else: 
            self.model = ReLUModel(Xs.shape[1], self.hidden_dim).to(device)


    def get_loss(self, detach=False): 
        Xs_prime = self.model(self.Xs)
        if not detach: 
            return (self.I * (Xs_prime - self.Xs)**2).sum()
        else: 
            return (self.I * (Xs_prime.detach() - self.Xs)**2).sum().item()

    def get_weights(self): 
        return self.model.linear1.weight.detach().cpu()

    def get_biases(self): 
        return rearrange(self.model.linear1.bias.detach(), 'd -> d 1').cpu()
    
    def forward(self, x): 
        return self.model(x)
    
    def train(self, epochs=100, batch_size=32, lr=1e-3, verbose=False): 
        self.model.train()
        self.Xs = repeat(self.Xs, 'n d -> (n b) d', b=batch_size)
        self.I = repeat(self.I, 'n d -> (n b) d', b=batch_size)
        epochs = epochs // batch_size
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            optimizer.zero_grad()
            loss = self.get_loss()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss: {loss.item():.4f}")

class DeepAutoEncoder(nn.Module):
    def __init__(self, Xs, I, mid_dims, hidden_dim, device='cpu'):
        super().__init__()

        self.Xs = Xs.to(device)
        self.I = repeat(I, 'd -> n d', n = Xs.shape[0]).to(device)
        self.hidden_dim = hidden_dim
        self.mid_dims = mid_dims

        encoder = []
        for i, mid_dim in enumerate(mid_dims): 
            if i == 0: 
                encoder.append(nn.Linear(Xs.shape[-1], mid_dim))
            else: 
                encoder.append(nn.Linear(mid_dims[i-1], mid_dim))
            encoder.append(nn.ReLU())
        encoder.append(nn.Linear(mid_dims[-1], hidden_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        for i, mid_dim in enumerate(mid_dims): 
            if i == 0: 
                decoder.append(nn.Linear(mid_dim, Xs.shape[-1]))
            else: 
                decoder.append(nn.Linear(mid_dim, mid_dims[i-1]))
            decoder.append(nn.ReLU())
        decoder.append(nn.Linear(hidden_dim, mid_dims[-1]))
        decoder = decoder[::-1]        
        self.decoder = nn.Sequential(*decoder)
        

    def forward(self, x, return_encoded=False): 
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        if return_encoded: 
            return decoded, encoded
        else: 
            return decoded
        
    def get_loss(self): 
        Xs_prime = self.forward(self.Xs)
        return (self.I * (Xs_prime - self.Xs)**2).sum()
    
    def train(self, epochs=100, batch_size=32, lr=1e-3, verbose=False): 
        self.Xs = repeat(self.Xs, 'n d -> (n b) d', b=batch_size)
        self.I = repeat(self.I, 'n d -> (n b) d', b=batch_size)
        epochs = epochs // batch_size
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            optimizer.zero_grad()
            loss = self.get_loss()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss: {loss.item():.4f}")

