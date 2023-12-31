

import imp
import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse

from GptConstants import GptConstants
from GPTLanguageModel import GPTLanguageModel
from ModelHandler import ModelHandler
device = 'cuda' if torch.cuda.is_available() else 'cpu'


model_handler: ModelHandler = ModelHandler()
model_handler.read_data()
model_handler.start_Gpt_model()
model_handler.load_model()
# model_handler.save_model()
print(model_handler.get_output("hello"))
















