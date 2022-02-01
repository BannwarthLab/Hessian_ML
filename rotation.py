from doctest import DocFileCase
from email import header
import pandas as pd
import numpy as np
import csv

df = pd.read_csv('coord.xyz',sep = '\t',header = 2)

data = df.to_numpy()
print(df)

alp = np.pi

vec = [1.0,0.,3.]

a = [[np.cos(alp),-np.sin(alp),0.],
     [np.sin(alp),np.cos(alp),0.],
     [0.,0.,1]]

print(np.matmul(a,vec))