from model_benchmark import  model_benchmark as model

import pandas as pd
import pytorch
import numpy
from torch.nn import BCELoss, BCEWithLogitsLoss, MSELoss

data = pd.read_csv("crime_processed.csv")
model = model()
