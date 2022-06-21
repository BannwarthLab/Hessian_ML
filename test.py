import numpy as np
import matplotlib.pyplot as plt

'''MAE = np.genfromtxt('MSE_list.txt')
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

plt.show()'''

MSE_het_0 = np.genfromtxt('MSE_list_het_0.txt')
MSE_het = np.genfromtxt('MSE_list_het.txt')
MSE_hom_0 = np.genfromtxt('MSE_list_hom_0.txt')
MSE_hom = np.genfromtxt('MSE_list_hom.txt')

MSE = [[MSE_het,MSE_hom],[MSE_het_0,MSE_hom_0]]

for i in range(2):
    if i == 0:
        plt.bar([i+1],[np.mean(MSE[i][0])],label='Heteronuclear')
        plt.bar([i+1],[np.mean(MSE[i][1])],label='Homonuclear')
    else:
        plt.bar(i+1,np.mean(MSE[i][0]))
        plt.bar(i+1,np.mean(MSE[i][1]))

plt.xticks([1,2],['1','2'])
plt.ylabel(r'$\bar{\mathrm{MSE}}$ [-]')
plt.legend()

plt.show()