import pandas as pd
import numpy as np
import csv




df = pd.DataFrame(['N',0.000000,0.00000,1.100000],
                  ['N',0.000000,0.00000,0.00000]).T
                  
with open('out.xyz', 'w') as f:
    # create the csv writer
    writer = csv.writer(f)
    print(len(df))
    # write a row to the csv file
    writer.writerow(f'{len(df)}')

df.to_csv('coord.xyz',mode = 'a',index = False)