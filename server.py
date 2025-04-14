import av
import sounddevice as sd

import itertools

import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import torch
from torch import nn
DEVICE=torch.device("cuda")
torch.set_default_device(DEVICE)#('cpu:0')

import torchaudio
from torchaudio.transforms import MelSpectrogram, Spectrogram, AmplitudeToDB, InverseMelScale, GriffinLim, InverseSpectrogram

from utils import *

import random

import pandas as pd

from models import *

from utils import *
import time
from multiprocessing.connection import Listener

#R1=torchaudio.transforms.Resample(44100,SR)
#R2=torchaudio.transforms.Resample(SR,44100)

print("initializing model")
model=(CombinedModel())
model.load_state_dict(torch.load("good_small_model2.pth"))
print("loaded model")
print("testing model...")
START=time.time()
for i in range(100):
    W0 = np.random.rand(4800,2).astype(np.float32)
    Wi = (torch.tensor(W0.T, device='cpu')).cuda()
    X = clamp(normalize(unwrap_complex(T(Wi))))
    with torch.no_grad():
        H = model(X)
    O = I(wrap_complex(denormalize(unclamp(H))))
    Wo = (O.cpu()).T.numpy()
END=time.time()
print("time to process 1000x4800x2 samples:",END-START)
print("averaged duration:",(END-START)/1000)
address = ('localhost', 6101)     # family is deduced to be 'AF_INET'
while True:
    try:
        with Listener(address) as listener:
            listener._listener._socket.settimeout(5)
            print("listening...")
            while True:
                with listener.accept() as conn:
                    print("got a connection!")
                    while True:
                        try:
                            X = conn.recv()
                        except:
                            print("got an error... closing connection...")
                            conn.close()
                            break
                        if isinstance(X,str) and X == 'close':
                            print("closing connection...")
                            conn.close()
                        X=torch.tensor(X).repeat(1,2).T
                        #print("X:",X.shape)
                        X = clamp(normalize(unwrap_complex((X))))
                        #print("X:",X.shape)
                        with torch.no_grad():
                            H = model(X)
                            #print("H:",H.shape)
                        O = (wrap_complex(denormalize(unclamp(H))))
                        #print("O:",O.shape)
                        O = O.T.cpu().numpy()
                        #print("O:",O.shape)
                        conn.send(O)
    except KeyboardInterrupt as e:
        print("exitting...")
        break
    except Exception as e:
        print("got error:",e)
        print("restarting listener...")
        time.sleep(0.1)