# %%
%load_ext autoreload
%autoreload 2
# %%
import torch as t
from featurevector import get_n_fvs
from EncoderDecoder import EncoderDecoder

# %%
# create a dataset of feature vectors
n = 1000 # number of feature vectors
dim = 25 # dimension of feature vectors
hdim = 5 # hidden dimension of encoder/decoder
S = 0.99 # sparsity
X = get_n_fvs(n, dim, S)
# importance vector 
I = t.linspace(1, 0, dim)


# %%
ed = EncoderDecoder(X, I, hdim, linear=True)
ed.train(epochs=100000, lr=1e-3)
edr = EncoderDecoder(X, I, hdim, linear=False)
edr.train(epochs=100000, lr=1e-3)
# %%
# visualize the weights on a grid side by side
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

W = ed.get_weights()
W = W.T @ W
Wr = edr.get_weights()
Wr = Wr.T @ Wr
b, br = ed.get_biases(), edr.get_biases()

print("Sparsity: ", S, "; Importance: decreasing linearly")
print("Epochs: ", 100000, "; Learning rate: ", 1e-3)
print("Linear loss: ", ed.get_loss(detach=True), "; ReLU loss: ", edr.get_loss(detach=True))

fig = make_subplots(rows=1, cols=4, subplot_titles=("Linear", "b", "ReLU", "b"), column_widths=[0.5, 0.05, 0.5, 0.05])
fig.add_trace(go.Heatmap(z=W, showscale=False, colorscale=['lightblue', 'whitesmoke', 'lightcoral'], zmin=-1, zmax=1), row=1, col=1)
fig.add_trace(go.Heatmap(z=b, showscale=False, colorscale=['lightblue','whitesmoke', 'lightcoral'], zmin=-1, zmax=1), row=1, col=2)
fig.add_trace(go.Heatmap(z=Wr, showscale=False, colorscale=['lightblue', 'whitesmoke', 'lightcoral'], zmin=-1, zmax=1), row=1, col=3)
fig.add_trace(go.Heatmap(z=br, showscale=False, colorscale=['lightblue', 'whitesmoke', 'lightcoral'], zmin=-1, zmax=1), row=1, col=4)
for i in range(1, 5): 
    fig.update_yaxes(autorange="reversed", row=1, col=i)
    if i % 2 == 0:
        fig.update_xaxes(visible=False, row=1, col=i)

fig.update_layout(height=500, width=1000, autosize=True, title_text="Linear vs ReLU")


# %%
