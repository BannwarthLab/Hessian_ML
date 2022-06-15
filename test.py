import numpy as np
import matplotlib.pyplot as plt

MAE = np.genfromtxt('MSE_list.txt')
MAE_L = []

for i in range(8):
    MS = 0
    for j in range(i,i+80,8):
        MS += MAE[j]*1/10
    MAE_L.append(MS)

print(MAE_L)

plt.plot(np.linspace(0.2,0.99,8),MAE_L,'x')
plt.xlabel('Training Set Fraction')
plt.ylabel(r'MAE of $k$ between mirrored Systems')

plt.savefig('MAE_Symmetry.png')
plt.savefig('MAE_Symmetry.svg')
plt.savefig('MAE_Symmetry.eps')

plt.show()