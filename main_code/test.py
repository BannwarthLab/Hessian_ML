from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from packages import *
from rotation_func import *

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
'''
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

plt.show()'''
cwd = os.getcwd()
cwd = cwd[:]+'/tests/main_test/'
mol_sys_dirs = sorted(os.listdir(cwd))[:-1]
h3_list = []
for i in range(10):
    h3_list.append(0.6+i*0.311111)

dist_arr = np.array([[0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25],
            [0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25],
            [0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7],
            [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0],
            h3_list,
            [0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25],
            [0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25],
            [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0],
            [0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25]
            ])

factor = np.array([2,2,2,0.4804+0.4804,1,2,2,1.12,2])
for i in range(len(dist_arr)):
    dist_arr[i] = np.array(dist_arr[i])*factor[i]


for mol in range(len(mol_sys_dirs)):
    #Gathering for each molecular systems all directories of diffrent structures
    mol_dir = f'{cwd}/{mol_sys_dirs[mol]}/'
    print(mol_sys_dirs[mol])
    struc_sys_dirs = sorted([ name for name in os.listdir(mol_dir) if os.path.isdir(os.path.join(mol_dir, name)) ])
    norm_list = [[],[],[]]
    for sys in range(len(struc_sys_dirs)):
        folder=f'{mol_dir}{struc_sys_dirs[sys]}/init_coord/'

        coord_file = f'{folder}coord.xyz'
        hessian_file = f'{folder}hessian'
        gradient_file = f'{folder}gradient'

        coord,header = import_coord(coord_file)

        hessian = import_hessian(hessian_file,coord_var=coord)


        gradient = import_gradient(gradient_file,coord_var=coord)

        hessian_approx = np.outer(gradient,gradient)
        n = 1
        norm_hessian_exact = linalg.norm(hessian)
        norm_hessian_approx = linalg.norm(hessian_approx)*n
        norm_hessian_diff = linalg.norm(hessian-hessian_approx)

        norm_list[0].append(norm_hessian_exact)
        norm_list[1].append(norm_hessian_approx)
        norm_list[2].append(norm_hessian_diff)

    fig,ax = plt.subplots()
    ax.plot(dist_arr[mol],norm_list[0],'x-',label=r'$||H^\mathrm{xTB}||_2$')
    ax.plot(dist_arr[mol],norm_list[1],'x-',label=r'$ || (g \otimes g)||_2$')
    ax.plot(dist_arr[mol],norm_list[2],'x-',label=r'$||H^\mathrm{xTB} - (g \otimes g)||_2$')
    ax.set(title = mol_sys_dirs[mol])
    ax.legend()
    ax.set_xlabel(r'$d$ [$\AA$]')
    ax.set_ylabel(r'$||X||_2$ [a.u.]')
    plt.savefig(f'main_code/plots/grad_test/{mol_sys_dirs[mol]}.svg')
    plt.savefig(f'main_code/plots/grad_test/{mol_sys_dirs[mol]}.png')
