import torch
import torchaudio
from torch import nn
import os
import sys

'''
                nn.Linear(window_size, window_size*2),
                nn.ELU(1),
                nn.Linear(window_size*2, window_size),
                nn.ELU(1),
                nn.Linear(window_size, 64),


                nn.Linear(64, window_size//2),
                nn.ELU(1),
                nn.Linear(window_size//2, window_size),
                nn.ELU(1),
                nn.Linear(window_size, window_size))



                nn.Linear(1024, 512, bias=False),
                nn.Linear(512, 128),
                nn.Linear(128, 512),
                nn.Linear(512,1024, bias=False))


                nn.Linear(window_size, 4096),
                nn.ReLU(4096),
                nn.Linear(4096, 2048),
                nn.Linear(2048, 64),
                nn.Linear(64, 512),
                nn.ReLU(512),
                nn.Linear(512, window_size))

                nn.Linear(window_size, 512),
                nn.Linear(512, 64),
                nn.Linear(64, 512),
                nn.Linear(512, window_size))
                '''

device = "mps"

if len(sys.argv) < 4:
    print(f"Usage: {sys.argv[0]} <model_state_file> <input_audio> <output_audio>")
    sys.exit(1)

state_file = sys.argv[1]
input_audio = sys.argv[2]
output_audio = sys.argv[3]

print("Loading state")
state = torch.load(state_file, map_location = torch.device('cpu'))

class NeuralNetwork_old(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
                nn.Linear(window_size, window_size*2),
                nn.ELU(1),
                nn.Linear(window_size*2, window_size),
                nn.ELU(1),
                nn.Linear(window_size, 64),

                nn.Linear(64, window_size//2),
                nn.ELU(1),
                nn.Linear(window_size//2, window_size),
                nn.ELU(1),
                nn.Linear(window_size, window_size))
    def forward(self, x):
        return self.stack(x)

class NeuralNetwork_a(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
                nn.Linear(1024, 512),
        #        nn.Tanh(),
                nn.Linear(512, 32),
        #        nn.Sigmoid(),
                nn.Linear(32, 512),
                nn.Linear(512,1024))
    def forward(self, x):
        return self.stack(x)


class NeuralNetwork_b(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
                nn.Linear(1024, 3000),
                nn.ELU(),
                nn.Linear(3000, 512),
                nn.ELU(),
                nn.Linear(512, 32),
                nn.ELU(),
                nn.Linear(32, 512),
                nn.ELU(),
                nn.Linear(512, 3000),
                nn.ELU(),
                nn.Linear(3000,1024))
    def forward(self, x):
        #x = self.flatten(x)
        return self.stack(x)

class NeuralNetwork_c(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
                nn.Linear(1024, 4096),
                nn.ELU(),
                nn.Linear(4096, 32),
                nn.ELU(),
                nn.Linear(32, 4096),
                nn.ELU(),
                nn.Linear(4096,1024))
    def forward(self, x):
        #x = self.flatten(x)
        return self.stack(x)


window_size = 2048
class NeuralNetwork_d(nn.Module):
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

                
window_size = 2048
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
                nn.Linear(window_size, 4096),
				nn.Sigmoid(),
                nn.Linear(4096, 2048),
                nn.Sigmoid(),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.Linear(256, 64),
                nn.Linear(64, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.Sigmoid(),
                nn.Linear(512, 1024),
                nn.Sigmoid(),
                nn.Linear(1024, window_size)) 
    def forward(self, x):
        #x = self.flatten(x)
        return self.stack(x)


model = NeuralNetwork().to(device)
print("Loading model")

model.load_state_dict(state['model_state_dict'])

model.eval()
print(f"Loading file {input_audio}")
half_size = window_size//2
waveform, samplerate = torchaudio.load(input_audio)
wf = torch.signal.windows.hann(window_size, sym=False, device=device)
zeros = torch.zeros(window_size, device=device)
hzeros = torch.zeros(half_size, device=device)
print("Transforming")
channels = []
for c in range(waveform.size(0)):
    half = hzeros
    channel = torch.tensor([], device=device)
    for w in range(0, waveform.size(1)-window_size, half_size):
        tw = None
        window = waveform[c][w:w+window_size].to(device)
        ampl = window.abs().max()
        if ampl.item() == 0.0:
            result = zeros
        else:
            result = model(window.divide(ampl).mul(wf)).mul(ampl)
        hresult, half = half.add(result[:half_size]), result[half_size:]
        #print(result.shape)
        channel = torch.cat((channel, hresult))
    # appending the ending half
    channel = torch.cat((channel, half))
    channels.append(channel)
channels = torch.vstack(channels)
print("Saving")

n = channels.to('cpu').detach()
torchaudio.save(output_audio, n, samplerate)
print("Done")
