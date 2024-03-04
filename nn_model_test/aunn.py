import torch
import torchaudio
from torch import nn
import os
import sys
from glob import glob

#device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device="mps"
liter='a'
window_size=2048
os.makedirs(liter, exist_ok=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
                nn.Linear(window_size, 4096),
				nn.ELU(0.1),
                nn.Linear(4096, 2048),
                nn.Linear(2048, 64),
				nn.ELU(0.1),
                nn.Linear(64, window_size))
    def forward(self, x):
        #x = self.flatten(x)
        return self.stack(x)

model = NeuralNetwork().to(device)
print(model)
print(device)

#loss_fn  = nn.L1Loss(reduction='mean')
#loss_fn = nn.MSELoss()
loss_fn = nn.SmoothL1Loss(beta=0.1)
#optimizer = torch.optim.Adadelta(model.parameters(), lr=0.7, weight_decay=0.1)
#optimizer = torch.optim.Adagrad(model.parameters())
#optimizer = torch.optim.ASGD(model.parameters())
optimizer = torch.optim.Adam(model.parameters(),  lr=0.7, weight_decay=0.1)
batch_size=64
window_step=1024
import random

def loader(batchsize=32, ws=0, step=2):
    batch = 0
    X =None  
    bs = 0
    l = glob(os.path.join(directory, '*.mp3'))
    random.shuffle(l)
    window_function = torch.signal.windows.hann(window_size, sym=False, device=device)
    for f in l:
        waveform, sample_rate = torchaudio.load(os.path.join(f), format="mp3")
        print("File '%s': shape: %s, sample rate: %d" % (f, waveform.shape, sample_rate))
        for w in range(ws, waveform.size(1)-window_size, step): # select window
            for c in range(waveform.size(0)): # for each window take channel
                
                window = waveform[c][w:w+window_size].to(device)
                ampl = window.abs().max()
                if ampl.item() == 0.0:
                    continue
                window = window.divide(ampl.item()).mul(window_function)
                X = torch.vstack((X, window)) if X is not None else window
                bs += 1
                if bs == batchsize:
                    yield X
                    bs = 0
                    X = None

directory = sys.argv[1]

model.eval()
def train(epoch, model, lossfn, optimizer):
    model.train()
    for batch, X in enumerate(loader(batch_size, epoch, 13)):
        #print(X.shape)
        #X = torch.tensor(X)
        X.to(device)
        pred = model(X)
        loss = lossfn(pred, X)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 1000 == 0:
            loss, current = loss.item(), (batch+1)*len(X)
            print(f"loss: {loss:>7f} [{current:>5d}])")

def test(epoch, model, lossfn):
    model.eval()
    test_loss, correct = 0, 0
    b = 0
    with torch.no_grad():
        for X in loader(128, epoch, 17):
            X = X.to(device)
            pred = model(X)
            test_loss += lossfn(pred, X).item()
            b += 1 
    print(f"Test loss: {test_loss/b:>7f}")
    return test_loss/b
epochs = 55
for t in range(epochs):
    print(f"Epoch {t} -------------------------------------------")
    train(t, model, loss_fn, optimizer)
    print(f"---------------------------------------------Testing {t}")
    loss = test(t, model,loss_fn)
    torch.save({
            'epoch': t,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, f"{liter}/epoch_state_{t}")
