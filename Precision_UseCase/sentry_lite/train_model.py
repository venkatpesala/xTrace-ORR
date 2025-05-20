# sentry_lite/train_model.py 

import pandas as pd 

# from sentry_lite.risk_model import train_model 
from risk_model import train_model

 

df = pd.read_excel("data/Synthetic Sponsor Risk Population -March 31 2025 -acb.xlsx") 

train_model(df) 