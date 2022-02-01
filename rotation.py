import pandas as pd
import numpy as np
import csv




df = pd.DataFrame([0.000000,0.00000,1.100000,'n'],
                  [0.000000,0.00000,0.00000,'n']).T
                  
with open('coord.xyz', 'w') as f:
    # create the csv writer
    writer = csv.writer(f)
    # write a row to the csv file
    writer.writerow(["$coord"])

df.to_csv('coord.xyz',mode = 'a',sep ='\t',index = False)

with open('coord.xyz', 'a') as f:
    # create the csv writer
    writer = csv.writer(f)
    # write a row to the csv file
    writer.writerow(["$end"])