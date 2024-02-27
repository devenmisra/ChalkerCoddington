# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: ccnet
#     language: python
#     name: python3
# ---

# +
import pickle as pkl
import pandas as pd

iP = 0.0

with open(f"completeLyapDataP{iP}.pickle", "rb") as f:
    object = pkl.load(f)
    
widths = [2, 4, 8, 16, 32, 64]

for width in widths: 

    df = pd.DataFrame(object[f'{width}'])
    df.to_csv(rf'lyapData_CSV/completeLyapDataM{width}P{iP}.csv')
