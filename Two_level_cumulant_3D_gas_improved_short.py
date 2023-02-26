#import libraries


import scipy
from scipy.integrate import solve_ivp
import numpy as np
import cmath as cm
import h5py
#from scipy.interpolate import UnivariateSpline
from numpy.linalg import multi_dot
from scipy.linalg import logm
from scipy.special import factorial
from scipy.special import *
#from scipy.signal import chirp, find_peaks, peak_widths
from scipy import sparse
from scipy.sparse import csr_matrix
from numpy.linalg import eig
from scipy.linalg import eig as sceig
import math
import time
#from math import comb
from sympy.physics.quantum.cg import CG
from sympy import S
import collections
import numpy.polynomial.polynomial as poly
import sys
argv=sys.argv

if len(argv) < 2:
    #Default
    run_id=1
else:
    try:
        run_id = int(argv[1])
        Natoms = int(argv[2])
        det_val_input = 0.0 #float(argv[3])
        
    except:
        print ("Input error")
        run_id=1

# some definitions (do not change!)

fe = 0
fg = 0

fixed_param = 2 # 0: L = 20, R = 0.5; 1: mean density; 2: optical depth along L.

# some definitions (do not change!)

e0 = np.array([0, 0, 1])
ex = np.array([1, 0, 0])
ey = np.array([0, 1, 0])
eplus = -(ex + 1j*ey)/np.sqrt(2)
eminus = (ex - 1j*ey)/np.sqrt(2)
single_decay = 1.0 # single atom decay rate

direc = '/data/rey/saag4275/data_files/'   # directory for saving data

# parameter setting box (may change)


realization_list = np.arange(1,101,1)
rabi_val_list = np.array([0.1, 0.5, 1.0]) #np.array([1.7,2.2,2.3,2.4,2.6,2.7]) #np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0])
tfin_list = np.array([40]*len(rabi_val_list)) #np.array([20]*2 + [8]*int(len(rabi_val_list)-2))
real_id_list = np.arange(0, len(realization_list), 1)
rabi_id_list = np.arange(0, len(rabi_val_list), 1)

# generate 2D array of realisation x rabi --> 

param_grid_rabi, param_grid_real = np.meshgrid(rabi_id_list, real_id_list, indexing='xy')
param_grid_real_list = param_grid_real.flatten()
param_grid_rabi_list = param_grid_rabi.flatten()

real_id = param_grid_real_list[run_id-1]
rabi_id = param_grid_rabi_list[run_id-1]

real_val = realization_list[real_id]
rabi_val = rabi_val_list[rabi_id] #rabi_val_list[run_id-1] 
t_final_input = tfin_list[rabi_id] 

print("Rabi = "+str(rabi_val), flush=True)
print("real id = "+str(real_val), flush=True)

eL = np.array([0, 0, 1]) # polarisation of laser, can be expressed in terms of the vectors defined above
detuning_list = np.array([0.0*single_decay]) # detuning of laser from transition
del_ze = 0.0 # magnetic field, i.e., Zeeman splitting of excited state manifold
del_zg = 0.0 # magnetic field, i.e., Zeeman splitting of ground state manifold
rabi = rabi_val*single_decay

#interactions turned off
turn_off_list = ['incoherent','coherent']
turn_off = [] #[turn_off_list[0], turn_off_list[1]] # leave turn_off = [], if nothing is to be turned off


turn_off_txt = ''
if turn_off != []:
    turn_off_txt += '_no_int_'
    for item in turn_off:
        turn_off_txt += '_'+ item

add_txt_in_params = turn_off_txt

num_pts_dr = int(5*1e2)

t_initial_dr = 0.0
t_final_dr = t_final_input 
t_range_dr = [t_initial_dr, t_final_dr]
t_vals_dr = np.linspace(t_initial_dr, t_final_dr, num_pts_dr) 


e0_desired = eL


# more definitions and functions (do not change!)

wavelength = 1 # wavelength of incident laser
k0 = 2*np.pi/wavelength
kvec = k0*np.array([1, 0, 0]) # k vector of incident laser


#choose which correlations should be cumulants

corr_list = ['sig_z.sig_z', 'sig_z.sig_p', 'sig_p.sig_p', 'sig_p.sig_m']
two_point_terms = corr_list #['sig_z.sig_p'] 

#################################################################################################

# more definitions and functions (do not change!)

def rotation_matrix_a_to_b(va, vb): #only works upto 1e15-ish precision
    ua = va/np.linalg.norm(va)
    ub = vb/np.linalg.norm(vb)
    if np.dot(ua, ub) == 1:
        return np.identity(3)
    elif np.dot(ua, ub) == -1: #changing z->-z changes y->-y, thus preserving x->x, which is the array direction (doesn't really matter though!)
        return -np.identity(3)
    uv = np.cross(ua,ub)
    c = np.dot(ua,ub)
    v_mat = np.zeros((3,3))
    ux = np.array([1,0,0])
    uy = np.array([0,1,0])
    uz = np.array([0,0,1])
    v_mat[:,0] = np.cross(uv, ux)
    v_mat[:,1] = np.cross(uv, uy)
    v_mat[:,2] = np.cross(uv, uz)
    matrix = np.identity(3) + v_mat + (v_mat@v_mat)*1.0/(1.0+c)
    return matrix

 
if np.abs(np.conj(e0)@e0_desired) < 1.0:
    rmat = rotation_matrix_a_to_b(e0,e0_desired)
    eplus = rmat@eplus
    eminus = rmat@eminus
    ex = rmat@ex
    ey = rmat@ey
    e0 = e0_desired

print('kL = '+str(kvec/np.linalg.norm(kvec)), flush=True)
print('e0 = '+str(e0), flush=True)
print('ex = '+str(ex), flush=True)
print('ey = '+str(ey), flush=True)
    
HSsize = int(2*fg + 1 + 2*fe + 1) # Hilbert space size of each atom
HSsize_tot = int(HSsize**Natoms) # size of total Hilbert space

adde = fe
addg = fg

# polarisation basis vectors
evec = {0: e0, 1:eplus, -1: eminus}
evec = collections.defaultdict(lambda : [0,0,0], evec) 
   
def sort_lists_simultaneously_cols(a, b): #a -list to be sorted, b - 2d array whose columns are to be sorted according to indices of a
    inds = a.argsort()
    sortedb = b[:,inds]
    return sortedb

# levels
deg_e = int(2*fe + 1)
deg_g = int(2*fg + 1)

if (deg_e == 1 and deg_g == 1):
    qmax = 0
else:
    qmax = 1

    
###########################################################################

# dictionaries

# MFT(m)/Cumulant(c) for each correlation

dict_corrs = {}
for i in range(0, len(two_point_terms)):
    dict_corrs[two_point_terms[i]] =  'c' 
dict_corrs = collections.defaultdict(lambda : 'm', dict_corrs) 

dict_corrs_10 = {}
for i in range(0, len(two_point_terms)):
    dict_corrs_10[two_point_terms[i]] =  1
dict_corrs_10 = collections.defaultdict(lambda : 0, dict_corrs_10) 

# Clebsch Gordan coeff
cnq = {}
arrcnq = np.zeros((deg_g, 2*qmax+1), complex)
if (deg_e == 1 and deg_g ==1):
    cnq[0, 0] = 1
    arrcnq[0, 0] =  1
else:
    for i in range(0, deg_g):
        mg = i-fg
        for q in range(-qmax, qmax+1):
            if np.abs(mg + q) <= fe:
                cnq[mg, q] =  np.float(CG(S(fg), S(mg), S(qmax), S(q), S(fe), S(mg+q)).doit())
                arrcnq[i, q+qmax] = cnq[mg, q]
cnq = collections.defaultdict(lambda : 0, cnq) 

# Dipole moment
dsph = {}
if (deg_e == 1 and deg_g ==1):
    dsph[0, 0] = np.conjugate(evec[0])
else:
    for i in range(0, deg_e):
        me = i-fe
        for j in range(0, deg_g):
            mg = j-fg
            dsph[me, mg] = (np.conjugate(evec[me-mg])*cnq[mg, me-mg])

dsph = collections.defaultdict(lambda : np.array([0,0,0]), dsph) 

# normalise vector
def hat_op(v):
    return (v/np.linalg.norm(v))

transition_wavelength = 1.0
R_perp_given = 0.5*transition_wavelength # radial std in units of lambda, experimental value for system
L_given = 20.0*transition_wavelength # axial std in units of lambda, experimental value for system
aspect_ratio = L_given/R_perp_given # = axial/radial std
N_given = 2000 # experimental value for system
k = 2*np.pi/transition_wavelength
OD_x_given = 3*N_given/(2*(k*R_perp_given)**2)
Volume_cloud_given = 2*np.pi*(R_perp_given**2)*L_given
mean_density_given = N_given/Volume_cloud_given

def f_cloud_dims_fixed_OD(N_output):
    # Since OD_x = 3*N_given/(2*(k*R_perp_given)**2)
    R_perp_output = R_perp_given*np.sqrt(N_output/N_given*1.0) # = np.sqrt(3*N_output/(2*OD_x*(k**2)))
    L_axial_output = R_perp_output*aspect_ratio

    return [R_perp_output, L_axial_output]


def f_cloud_dims_fixed_mean_density(N_output):
    Volume_cloud_output = N_output/mean_density_given
    R_perp_output = (Volume_cloud_output/(2*np.pi*aspect_ratio))**(1/3.0)
    L_axial_output = R_perp_output*aspect_ratio

    return [R_perp_output, L_axial_output]

if fixed_param == 0:
    std_list = [R_perp_given, L_given]
    fixed_text = '_fixed_size'
elif fixed_param == 1:
    std_list = f_cloud_dims_fixed_mean_density(Natoms)
    fixed_text = '_fixed_mean_density'
elif fixed_param == 2:
    std_list = f_cloud_dims_fixed_OD(Natoms)
    fixed_text = '_fixed_OD_ax'
    
std_rad = std_list[0]
std_ax = std_list[1]
dims_text = '_std_rad_'+str(np.round(std_rad,2)).replace('.',',') + '_std_ax_'+str(np.round(std_ax,2)).replace('.',',')

# plot properties

levels = int(deg_e + deg_g)
rdir_fig = '_3D_gas'+fixed_text+dims_text

eLdir_fig = '_eL_along_'
dirs = ['x','y','z']
temp_add = 0
for i in range(0,3):
    if eL[i]!=0:
        if temp_add == 0:
            eLdir_fig += dirs[i]
        else:
            eLdir_fig += '_and_'+ dirs[i]
        temp_add += 1

kdir_fig = '_k_along_'
dirs = ['x','y','z']
temp_add = 0
for i in range(0,3):
    if kvec[i]!=0:
        if temp_add == 0:
            kdir_fig += dirs[i]
        else:
            kdir_fig += '_and_'+ dirs[i]
        temp_add += 1

rabi_add = '_rabi_'+str((rabi)).replace('.',',')


h5_title = str(levels)+'_level_'+str(Natoms)+'_atoms'+rdir_fig+kdir_fig+eLdir_fig+'_real_id_'+str(int(real_val))+'.h5'

h5_title_dr = str(levels)+'_level_'+str(Natoms)+'_atoms'+rdir_fig+kdir_fig+eLdir_fig+rabi_add+'_tfin_'+str(int(t_final_input))+'_real_id_'+str(int(real_val))+add_txt_in_params+'.h5'


# try to load data for positions, if no data available, then generate samples

try:
    hf = h5py.File(direc+'Atomic_positions_Greens_fn_phases_3D_gas_'+h5_title, 'r')

    rvecall = hf['rvecall'][()]
    arrGij = hf['arrGij'][()]
    arrGijtilde = hf['arrGijtilde'][()]
    arrIij = hf['arrIij'][()]
    phase_array = hf['forward_phase_array'][()]
    hf.close()
    
    print("Data for positions loaded from file!", flush=True)




except:

    print("No data for positions found, will generate sampling of positions now and save it!", flush=True)

    temp = np.random.default_rng(real_val)

    r_sampled_raw = temp.normal(0, std_ax, Natoms*1000) # get axial x position
    r_sampled = temp.choice(r_sampled_raw, Natoms, replace=False) # to prevent multiple atoms from being at exactly the same positions
    r_array_x = np.sort(r_sampled)

    r_sampled_rad_z = temp.normal(0, std_rad, Natoms)  # get radial z position
    r_array_z = r_sampled_rad_z 

    r_sampled_rad_y = temp.normal(0, std_rad, Natoms)  # get radial y position
    r_array_y = r_sampled_rad_y 

    r_array_xyz = np.array([r_array_x, r_array_y, r_array_z])
    rvecall = r_array_xyz.T

    r_nn_spacing = np.sqrt(np.einsum('ab->b', (r_array_xyz[:,1:] - r_array_xyz[:,:-1])**2))

    r_nn_spacing_avg = np.mean(r_nn_spacing) # 3d avg spacing for simulation
    r_min = np.min(r_nn_spacing) # 3d min spacing for simulation

    print('3D gas properties:' , flush=True)
    print('std axial = ' + str(std_ax))
    print('std radial = ' + str(std_rad), flush=True)
    print('mean nearest-neighbor spacing = ' + str(r_nn_spacing_avg), flush=True)
    print('minimum nearest-neighbor spacing = ' + str(r_min), flush=True)
    
    
    # phase_array
    
    phase_array = np.zeros((Natoms, Natoms), complex)
    for i in range(0, Natoms):
        for j in range(0, Natoms):
            temp_phase = np.dot(kvec, (rvecall[i] - rvecall[j]))
            phase_array[i, j] = np.exp(1j*temp_phase)

    # Green's function
    def funcG(r):
        tempcoef = 3*single_decay/4.0
        temp1 = (np.identity(3) - np.outer(hat_op(r), hat_op(r)))*np.exp(1j*k0*np.linalg.norm(r))/(k0*np.linalg.norm(r)) 
        temp2 = (np.identity(3) - 3*np.outer(hat_op(r), hat_op(r)))*((1j*np.exp(1j*k0*np.linalg.norm(r))/(k0*np.linalg.norm(r))**2) - np.exp(1j*k0*np.linalg.norm(r))/(k0*np.linalg.norm(r))**3)
        return (tempcoef*(temp1 + temp2))

    def funcGij(i, j):
        return (funcG(rvecall[i] - rvecall[j]))

    fac_inc = 1.0
    fac_coh = 1.0
    if turn_off!=[]:
        for item in turn_off:
            if item == 'incoherent':
                fac_inc = 0
            if item == 'coherent':
                fac_coh = 0


    taD = time.time()

    dictRij = {}
    dictIij = {}
    dictGij = {}
    dictGijtilde = {}

    for i in range(0, Natoms):
        for j in range(0, Natoms):
            for q1 in range(-qmax,qmax+1):
                for q2 in range(-qmax,qmax+1):
                    if i!=j:
                        tempRij = fac_coh*np.conjugate(evec[q1])@np.real(funcGij(i, j))@evec[q2]
                        tempIij = fac_inc*np.conjugate(evec[q1])@np.imag(funcGij(i, j))@evec[q2]

                    else:
                        tempRij = 0
                        tempIij = (single_decay/2.0)*np.dot(np.conjugate(evec[q1]),evec[q2])
                    dictRij[i, j, q1, q2] = tempRij
                    dictIij[i, j, q1, q2] = tempIij
                    dictGij[i, j, q1, q2] = tempRij + 1j*tempIij
                    dictGijtilde[i, j, q1, q2] = tempRij - 1j*tempIij
                    #arrGij[i, j, q1+qmax, q2+qmax] = tempRij + 1j*tempIij
                    #arrGijtilde[i, j, q1+qmax, q2+qmax] = tempRij - 1j*tempIij

    dictRij = collections.defaultdict(lambda : 0, dictRij) 
    dictIij = collections.defaultdict(lambda : 0, dictIij) 
    dictGij = collections.defaultdict(lambda : 0, dictGij) 
    dictGijtilde = collections.defaultdict(lambda : 0, dictGijtilde) 

    tbD = time.time()
    print("time to assign Rij, Iij dict: "+str(tbD-taD), flush=True)

    taG = time.time()

    arrGij = np.zeros((Natoms, Natoms, deg_e, deg_g, deg_e, deg_g), complex)
    arrGijtilde = np.zeros((Natoms, Natoms, deg_e, deg_g, deg_e, deg_g), complex)
    arrIij = np.zeros((Natoms, Natoms, deg_e, deg_g, deg_e, deg_g), complex)
    for i in range(0, Natoms):
        for j in range(0, Natoms):
            for ima in range(0, deg_e):
                ma = ima - fe
                for ina in range(0, deg_g):
                    na = ina - fg
                    for imb in range(0, deg_e):
                        mb = imb - fe
                        for inb in range(0, deg_g):
                            nb = inb - fg
                            arrGij[i, j, ima, ina, imb, inb] = dictGij[i, j, ma-na, mb-nb]*cnq[na, ma-na]*cnq[nb, mb-nb]
                            arrGijtilde[i, j, ima, ina, imb, inb] = dictGijtilde[i, j, ma-na, mb-nb]*cnq[na, ma-na]*cnq[nb, mb-nb]
                            arrIij[i, j, ima, ina, imb, inb] = dictIij[i, j, ma-na, mb-nb]*cnq[na, ma-na]*cnq[nb, mb-nb]

    tbG = time.time()
    print("time to assign Gij matrix: "+str(tbG-taG), flush=True)



    # save position and Gij data for future reference
    
    if fac_coh == 1.0 and fac_inc == 1.0:
        hf = h5py.File(direc+'Atomic_positions_Greens_fn_phases_3D_gas_'+h5_title, 'w')
        hf.create_dataset('rvecall', data=rvecall, compression="gzip", compression_opts=9)
        hf.create_dataset('arrGij', data=arrGij, compression="gzip", compression_opts=9)
        hf.create_dataset('arrGijtilde', data=arrGijtilde, compression="gzip", compression_opts=9)
        hf.create_dataset('arrIij', data=arrIij, compression="gzip", compression_opts=9)
        hf.create_dataset('forward_phase_array', data=phase_array, compression="gzip", compression_opts=9)
        hf.close()
        
    else:
        hf = h5py.File(direc+'Atomic_positions_Greens_fn_phases_3D_gas_'+h5_title_turn_off, 'w')
        hf.create_dataset('rvecall', data=rvecall, compression="gzip", compression_opts=9)
        hf.create_dataset('arrGij', data=arrGij, compression="gzip", compression_opts=9)
        hf.create_dataset('arrGijtilde', data=arrGijtilde, compression="gzip", compression_opts=9)
        hf.create_dataset('arrIij', data=arrIij, compression="gzip", compression_opts=9)
        hf.create_dataset('forward_phase_array', data=phase_array, compression="gzip", compression_opts=9)
        hf.close()
    
    if fac_coh != 1.0 and fac_inc == 1.0:
            print("Coherent interactions turned off!", flush=True)
    elif fac_coh == 1.0 and fac_inc != 1.0:
            print("Incoherent interactions turned off!", flush=True)
    elif fac_coh != 1.0 and fac_inc != 1.0:
            print("Coherent AND incoherent interactions turned off!", flush=True)



# redifining to keep q = 0 only

arrGij = arrGij[:,:,0,0,0,0]
arrGijtilde = arrGijtilde[:,:,0,0,0,0]
arrIij = arrIij[:,:,0,0,0,0]


#Rabi frequency for each atom


omega_atom = (rabi*np.dot(dsph[0, 0],eL)*np.exp(1j*np.einsum('x,nx->n', kvec, rvecall))) 
            

#############################################################################################################

# don't change


# single atom operators' index + correlations' index -- in the list of all ops for all atoms

dict_ops = {}
index = 0
for n in range(0, Natoms):

    dict_ops['z', n] = index
    index += 1
    
for n in range(0, Natoms):

    dict_ops['p', n] = index
    index += 1

for n1 in range(0, Natoms):
    for n2 in range(0, Natoms):  #correlations of the form sig_ab^k . sig_cd^l 

        dict_ops['z,z', n1, n2] = index
        index += 1
        
for n1 in range(0, Natoms):
    for n2 in range(0, Natoms):  #correlations of the form sig_ab^k . sig_cd^l 


        dict_ops['z,p', n1, n2] = index
        dict_ops['p,z', n2, n1] = index
        index += 1
        
for n1 in range(0, Natoms):
    for n2 in range(0, Natoms):  #correlations of the form sig_ab^k . sig_cd^l 


        dict_ops['p,p', n1, n2] = index
        index += 1
        
for n1 in range(0, Natoms):
    for n2 in range(0, Natoms):  #correlations of the form sig_ab^k . sig_cd^l 


        dict_ops['p,m', n1, n2] = index
        dict_ops['m,p', n2, n1] = index
        index += 1
        
dict_ops = collections.defaultdict(lambda : 'None', dict_ops)

####################################################################################################

# (1-identity) array to prevent counting of i = j terms (remove self-interaction terms)

delta_ij = 1 - np.identity(Natoms)

# 3d array (i,j,k) that is equal to zero for i = j or j = k or i = k

theta_ijk = np.zeros((Natoms, Natoms, Natoms))

for i in range(0, Natoms):
    for j in range(0, Natoms):
        if i!=j:
            for k in range(0, Natoms):
                if j!=k and i!=k:
                    theta_ijk[i,j,k] = 1

#EOM for one point functions

def f_sig_z_dot(sig_z, sig_p, sig_p_sig_m, drive): #n = 0, 1, 2, ... = atom no., drive = 0, 1
    #free index n
    tempsum = 2*drive*1j*(sig_p*omega_atom - np.conjugate(omega_atom)*np.conjugate(sig_p)) - (sig_z + 1)*single_decay    
    tempsum += dict_corrs_10['sig_p.sig_m']*(2*1j*np.einsum('nj,nj,nj->n', delta_ij, arrGij,sig_p_sig_m)) + (1-dict_corrs_10['sig_p.sig_m'])*(2*1j*np.einsum('nj,nj,n,j->n', delta_ij, arrGij,sig_p,np.conj(sig_p)))
    tempsum += dict_corrs_10['sig_p.sig_m']*(-2*1j*np.einsum('jn,jn,jn->n', delta_ij, arrGijtilde,sig_p_sig_m)) + (1-dict_corrs_10['sig_p.sig_m'])*(-2*1j*np.einsum('jn,jn,j,n->n', delta_ij, arrGijtilde,sig_p,np.conj(sig_p)))
    return tempsum

def f_sig_p_dot(sig_z, sig_p, sig_z_sig_p, detuning, drive): #n = 0, 1, 2, ... = atom no.
    tempsum = (1j*detuning - 0.5*single_decay)*sig_p + drive*1j*sig_z*np.conjugate(omega_atom) 
    tempsum += (dict_corrs_10['sig_z.sig_p']*1j*np.einsum('nj,nj,nj->n', delta_ij, arrGijtilde, sig_z_sig_p)) + ((1-dict_corrs_10['sig_z.sig_p'])*1j*np.einsum('nj,nj,n,j->n', delta_ij, arrGijtilde, sig_z, sig_p))
    return tempsum


################################################################################

#EOM for two point functions

def f_sig_z_sig_z_dot(sig_z, sig_p, sig_z_sig_z, sig_z_sig_p, sig_p_sig_m, detuning, drive): #n = 0, 1, 2, ... = atom no.

    sig_z_i, sig_z_j = np.meshgrid(sig_z, sig_z, indexing='ij')

    # driven part
    tempsum = dict_corrs_10['sig_z.sig_p']*drive*2*1j*(np.einsum('ij,i,ji->ij', delta_ij,omega_atom,sig_z_sig_p) + np.einsum('ij,j,ij->ij', delta_ij,omega_atom,sig_z_sig_p)) - dict_corrs_10['sig_z.sig_p']*drive*2*1j*(np.einsum('ij,i,ji->ij', delta_ij,np.conj(omega_atom),np.conj(sig_z_sig_p)) + np.einsum('ij,j,ij->ij', delta_ij,np.conj(omega_atom),np.conj(sig_z_sig_p)))
    tempsum += (1-dict_corrs_10['sig_z.sig_p'])*drive*2*1j*(np.einsum('ij,i,j,i->ij', delta_ij,omega_atom,sig_z,sig_p) + np.einsum('ij,j,i,j->ij', delta_ij,omega_atom,sig_z,sig_p)) - (1-dict_corrs_10['sig_z.sig_p'])*drive*2*1j*(np.einsum('ij,i,j,i->ij', delta_ij,np.conj(omega_atom),sig_z,np.conj(sig_p)) + np.einsum('ij,j,i,j->ij', delta_ij,np.conj(omega_atom),sig_z,np.conj(sig_p)))
    
    # decay part
    tempsum += -(2*sig_z_sig_z + sig_z_i + sig_z_j)*single_decay

    # interacting part (two-point corrs)
    tempsum += dict_corrs_10['sig_p.sig_m']*4*np.einsum('ij,ij,ij->ij', delta_ij, arrIij, sig_p_sig_m + np.conj(sig_p_sig_m))
    tempsum += (1-dict_corrs_10['sig_p.sig_m'])*4*(np.einsum('ij,ij,i,j->ij', delta_ij, arrIij, sig_p, np.conj(sig_p)) + np.einsum('ij,ij,j,i->ij', delta_ij, arrIij, sig_p, np.conj(sig_p)))

    # interacting part (three-point corrs)
    #1
    temp1 = 2*1j*np.einsum('ijk,ik,j,i,k->ij', theta_ijk, arrGij, sig_z, sig_p, np.conj(sig_p))
    tempsum += dict_corrs_10['sig_z.sig_p']*2*1j*np.einsum('ijk,ik,ji,k->ij', theta_ijk, arrGij, sig_z_sig_p, np.conj(sig_p))
    tempsum += dict_corrs_10['sig_p.sig_m']*2*1j*np.einsum('ijk,ik,ik,j->ij', theta_ijk, arrGij, sig_p_sig_m, sig_z)
    tempsum += dict_corrs_10['sig_z.sig_p']*2*1j*np.einsum('ijk,ik,jk,i->ij', theta_ijk, arrGij, np.conj(sig_z_sig_p), sig_p)
    tempsum += (1-dict_corrs_10['sig_z.sig_p'])*temp1 + (1-dict_corrs_10['sig_p.sig_m'])*temp1  + (1-dict_corrs_10['sig_z.sig_p'])*temp1 
    tempsum += -2*temp1

    #2
    temp2 = -2*1j*np.einsum('ijk,ik,j,k,i->ij', theta_ijk, arrGijtilde, sig_z, sig_p, np.conj(sig_p))
    tempsum += -dict_corrs_10['sig_z.sig_p']*2*1j*np.einsum('ijk,ik,jk,i->ij', theta_ijk, arrGijtilde, sig_z_sig_p, np.conj(sig_p))
    tempsum += -dict_corrs_10['sig_p.sig_m']*2*1j*np.einsum('ijk,ik,ki,j->ij', theta_ijk, arrGijtilde, sig_p_sig_m, sig_z)
    tempsum += -dict_corrs_10['sig_z.sig_p']*2*1j*np.einsum('ijk,ik,ji,k->ij', theta_ijk, arrGijtilde, np.conj(sig_z_sig_p), sig_p)
    tempsum += (1-dict_corrs_10['sig_z.sig_p'])*temp2 + (1-dict_corrs_10['sig_p.sig_m'])*temp2  + (1-dict_corrs_10['sig_z.sig_p'])*temp2 
    tempsum += -2*temp2

    #3
    temp3 = 2*1j*np.einsum('ijk,jk,i,j,k->ij', theta_ijk, arrGij, sig_z, sig_p, np.conj(sig_p))
    tempsum += dict_corrs_10['sig_z.sig_p']*2*1j*np.einsum('ijk,jk,ij,k->ij', theta_ijk, arrGij, sig_z_sig_p, np.conj(sig_p))
    tempsum += dict_corrs_10['sig_p.sig_m']*2*1j*np.einsum('ijk,jk,jk,i->ij', theta_ijk, arrGij, sig_p_sig_m, sig_z)
    tempsum += dict_corrs_10['sig_z.sig_p']*2*1j*np.einsum('ijk,jk,ik,j->ij', theta_ijk, arrGij, np.conj(sig_z_sig_p), sig_p)
    tempsum += (1-dict_corrs_10['sig_z.sig_p'])*temp3 + (1-dict_corrs_10['sig_p.sig_m'])*temp3  + (1-dict_corrs_10['sig_z.sig_p'])*temp3 
    tempsum += -2*temp3

    #4
    temp4 = -2*1j*np.einsum('ijk,jk,i,k,j->ij', theta_ijk, arrGijtilde, sig_z, sig_p, np.conj(sig_p))
    tempsum += -dict_corrs_10['sig_z.sig_p']*2*1j*np.einsum('ijk,jk,ik,j->ij', theta_ijk, arrGijtilde, sig_z_sig_p, np.conj(sig_p))
    tempsum += -dict_corrs_10['sig_p.sig_m']*2*1j*np.einsum('ijk,jk,kj,i->ij', theta_ijk, arrGijtilde, sig_p_sig_m, sig_z)
    tempsum += -dict_corrs_10['sig_z.sig_p']*2*1j*np.einsum('ijk,jk,ij,k->ij', theta_ijk, arrGijtilde, np.conj(sig_z_sig_p), sig_p)
    tempsum += (1-dict_corrs_10['sig_z.sig_p'])*temp4 + (1-dict_corrs_10['sig_p.sig_m'])*temp4  + (1-dict_corrs_10['sig_z.sig_p'])*temp4 
    tempsum += -2*temp4


    return tempsum


def f_sig_z_sig_p_dot(sig_z, sig_p, sig_z_sig_z, sig_z_sig_p, sig_p_sig_m, sig_p_sig_p, detuning, drive): #n = 0, 1, 2, ... = atom no.

    sig_p_i, sig_p_j = np.meshgrid(sig_p, sig_p, indexing='ij')

    # driven part
    tempsum = drive*2*1j*(dict_corrs_10['sig_p.sig_p']*np.einsum('ij,i,ij->ij', delta_ij,omega_atom,sig_p_sig_p)) - drive*2*1j*(dict_corrs_10['sig_p.sig_m']*np.einsum('ij,i,ji->ij', delta_ij,np.conj(omega_atom),sig_p_sig_m)) + drive*1j*dict_corrs_10['sig_z.sig_z']*np.einsum('ij,j,ij->ij', delta_ij,np.conj(omega_atom),sig_z_sig_z)
    tempsum += drive*2*1j*((1-dict_corrs_10['sig_p.sig_p'])*np.einsum('ij,i,i,j->ij', delta_ij,omega_atom,sig_p,sig_p)) - drive*2*1j*((1-dict_corrs_10['sig_p.sig_m'])*np.einsum('ij,i,j,i->ij', delta_ij,np.conj(omega_atom),sig_p,np.conj(sig_p))) + drive*1j*(1-dict_corrs_10['sig_z.sig_z'])*np.einsum('ij,j,i,j->ij', delta_ij,np.conj(omega_atom),sig_z,sig_z)
    
    # decay part
    tempsum += -sig_p_j*single_decay + sig_z_sig_p*(-3*single_decay/2.0 + 1j*detuning)

    # interacting part (two-point corrs)
    tempsum += dict_corrs_10['sig_z.sig_p']*2*np.einsum('ij,ij,ji->ij', delta_ij, arrIij, sig_z_sig_p) + 1j*np.einsum('ij,i,ij->ij', delta_ij,sig_p,arrGij)
    tempsum += (1-dict_corrs_10['sig_z.sig_p'])*2*np.einsum('ij,ij,j,i->ij', delta_ij, arrIij, sig_z, sig_p) 

    # interacting part (three-point corrs)
    #1
    temp1 = 2*1j*np.einsum('ijk,ik,j,i,k->ij', theta_ijk, arrGij, sig_p, sig_p, np.conj(sig_p))
    tempsum += dict_corrs_10['sig_p.sig_p']*2*1j*np.einsum('ijk,ik,ji,k->ij', theta_ijk, arrGij, sig_p_sig_p, np.conj(sig_p))
    tempsum += dict_corrs_10['sig_p.sig_m']*2*1j*np.einsum('ijk,ik,ik,j->ij', theta_ijk, arrGij, sig_p_sig_m, sig_p)
    tempsum += dict_corrs_10['sig_p.sig_m']*2*1j*np.einsum('ijk,ik,jk,i->ij', theta_ijk, arrGij, sig_p_sig_m, sig_p)
    tempsum += (1-dict_corrs_10['sig_p.sig_p'])*temp1 + (1-dict_corrs_10['sig_p.sig_m'])*temp1  + (1-dict_corrs_10['sig_p.sig_m'])*temp1 
    tempsum += -2*temp1

    #2
    temp2 = -2*1j*np.einsum('ijk,ik,j,k,i->ij', theta_ijk, arrGijtilde, sig_p, sig_p, np.conj(sig_p))
    tempsum += -dict_corrs_10['sig_p.sig_p']*2*1j*np.einsum('ijk,ik,jk,i->ij', theta_ijk, arrGijtilde, sig_p_sig_p, np.conj(sig_p))
    tempsum += -dict_corrs_10['sig_p.sig_m']*2*1j*np.einsum('ijk,ik,ki,j->ij', theta_ijk, arrGijtilde, sig_p_sig_m, sig_p)
    tempsum += -dict_corrs_10['sig_p.sig_m']*2*1j*np.einsum('ijk,ik,ji,k->ij', theta_ijk, arrGijtilde, sig_p_sig_m, sig_p)
    tempsum += (1-dict_corrs_10['sig_p.sig_p'])*temp2 + (1-dict_corrs_10['sig_p.sig_m'])*temp2  + (1-dict_corrs_10['sig_p.sig_m'])*temp2 
    tempsum += -2*temp2

    #3
    temp3 = 1j*np.einsum('ijk,jk,j,k,i->ij', theta_ijk, arrGijtilde, sig_z, sig_p, sig_z)
    tempsum += dict_corrs_10['sig_z.sig_p']*1j*np.einsum('ijk,jk,jk,i->ij', theta_ijk, arrGijtilde, sig_z_sig_p, sig_z)
    tempsum += dict_corrs_10['sig_z.sig_p']*1j*np.einsum('ijk,jk,ik,j->ij', theta_ijk, arrGijtilde, sig_z_sig_p, sig_z)
    tempsum += dict_corrs_10['sig_z.sig_z']*1j*np.einsum('ijk,jk,ij,k->ij', theta_ijk, arrGijtilde, sig_z_sig_z, sig_p)
    tempsum += (1-dict_corrs_10['sig_z.sig_p'])*temp3 + (1-dict_corrs_10['sig_z.sig_p'])*temp3  + (1-dict_corrs_10['sig_z.sig_z'])*temp3 
    tempsum += -2*temp3


    return tempsum


def f_sig_p_sig_p_dot(sig_z, sig_p, sig_z_sig_p, sig_p_sig_p, detuning, drive): #n = 0, 1, 2, ... = atom no.

    # driven part
    tempsum = drive*1j*(dict_corrs_10['sig_z.sig_p']*np.einsum('ij,i,ij->ij', delta_ij,np.conj(omega_atom),sig_z_sig_p)) + drive*1j*(dict_corrs_10['sig_z.sig_p']*np.einsum('ij,j,ji->ij', delta_ij,np.conj(omega_atom),sig_z_sig_p))
    tempsum += drive*1j*((1-dict_corrs_10['sig_z.sig_p'])*np.einsum('ij,i,i,j->ij', delta_ij,np.conj(omega_atom),sig_z,sig_p)) + drive*1j*((1-dict_corrs_10['sig_z.sig_p'])*np.einsum('ij,j,j,i->ij', delta_ij,np.conj(omega_atom),sig_z,sig_p))

    # decay part
    tempsum += -sig_p_sig_p*(single_decay - 2*1j*detuning)

    # interacting part (three-point corrs)
    #1
    temp1 = 1j*np.einsum('ijk,ik,i,j,k->ij', theta_ijk, arrGijtilde, sig_z, sig_p, sig_p)
    tempsum += dict_corrs_10['sig_p.sig_p']*1j*np.einsum('ijk,ik,jk,i->ij', theta_ijk, arrGijtilde, sig_p_sig_p, sig_z)
    tempsum += dict_corrs_10['sig_z.sig_p']*1j*np.einsum('ijk,ik,ik,j->ij', theta_ijk, arrGijtilde, sig_z_sig_p, sig_p)
    tempsum += dict_corrs_10['sig_z.sig_p']*1j*np.einsum('ijk,ik,ij,k->ij', theta_ijk, arrGijtilde, sig_z_sig_p, sig_p)
    tempsum += (1-dict_corrs_10['sig_p.sig_p'])*temp1 + (1-dict_corrs_10['sig_z.sig_p'])*temp1  + (1-dict_corrs_10['sig_z.sig_p'])*temp1 
    tempsum += -2*temp1

    #2
    temp2 = 1j*np.einsum('ijk,jk,j,i,k->ij', theta_ijk, arrGijtilde, sig_z, sig_p, sig_p)
    tempsum += dict_corrs_10['sig_p.sig_p']*1j*np.einsum('ijk,jk,ik,j->ij', theta_ijk, arrGijtilde, sig_p_sig_p, sig_z)
    tempsum += dict_corrs_10['sig_z.sig_p']*1j*np.einsum('ijk,jk,jk,i->ij', theta_ijk, arrGijtilde, sig_z_sig_p, sig_p)
    tempsum += dict_corrs_10['sig_z.sig_p']*1j*np.einsum('ijk,jk,ji,k->ij', theta_ijk, arrGijtilde, sig_z_sig_p, sig_p)
    tempsum += (1-dict_corrs_10['sig_p.sig_p'])*temp2 + (1-dict_corrs_10['sig_z.sig_p'])*temp2  + (1-dict_corrs_10['sig_z.sig_p'])*temp2 
    tempsum += -2*temp2


    return tempsum


def f_sig_p_sig_m_dot(sig_z, sig_p, sig_z_sig_z, sig_z_sig_p, sig_p_sig_m, sig_p_sig_p, detuning, drive):  #n = 0, 1, 2, ... = atom no.

    # driven part
    tempsum = drive*1j*(-dict_corrs_10['sig_z.sig_p']*np.einsum('ij,j,ji->ij', delta_ij,omega_atom,sig_z_sig_p)) + drive*1j*(dict_corrs_10['sig_z.sig_p']*np.einsum('ij,i,ij->ij', delta_ij,np.conj(omega_atom),np.conj(sig_z_sig_p)))
    tempsum += drive*1j*(-(1-dict_corrs_10['sig_z.sig_p'])*np.einsum('ij,j,j,i->ij', delta_ij,omega_atom,sig_z,sig_p)) + drive*1j*((1-dict_corrs_10['sig_z.sig_p'])*np.einsum('ij,i,i,j->ij', delta_ij,np.conj(omega_atom),sig_z,np.conj(sig_p)))

    # decay part
    tempsum += -sig_p_sig_m*single_decay + dict_corrs_10['sig_z.sig_z']*np.einsum('ij,ij,ij->ij', delta_ij,sig_z_sig_z,arrIij) + (1-dict_corrs_10['sig_z.sig_z'])*np.einsum('ij,i,j,ij->ij', delta_ij,sig_z,sig_z,arrIij)

    # interacting part (one-point terms)
    tempsum += 0.5*1j*np.einsum('ij,ij,i->ij', delta_ij, arrGijtilde, sig_z) - 0.5*1j*np.einsum('ij,ij,j->ij', delta_ij, arrGij, sig_z)

    # interacting part (three-point corrs)
    #1
    temp1 = -1j*np.einsum('ijk,jk,j,i,k->ij', theta_ijk, arrGij, sig_z, sig_p, np.conj(sig_p))
    tempsum += -dict_corrs_10['sig_z.sig_p']*1j*np.einsum('ijk,jk,ji,k->ij', theta_ijk, arrGij, sig_z_sig_p, np.conj(sig_p))
    tempsum += -dict_corrs_10['sig_p.sig_m']*1j*np.einsum('ijk,jk,ik,j->ij', theta_ijk, arrGij, sig_p_sig_m, sig_z)
    tempsum += -dict_corrs_10['sig_z.sig_p']*1j*np.einsum('ijk,jk,jk,i->ij', theta_ijk, arrGij, np.conj(sig_z_sig_p), sig_p)
    tempsum += (1-dict_corrs_10['sig_z.sig_p'])*temp1 + (1-dict_corrs_10['sig_p.sig_m'])*temp1  + (1-dict_corrs_10['sig_z.sig_p'])*temp1 
    tempsum += -2*temp1

    #2
    temp2 = 1j*np.einsum('ijk,ik,i,k,j->ij', theta_ijk, arrGijtilde, sig_z, sig_p, np.conj(sig_p))
    tempsum += dict_corrs_10['sig_z.sig_p']*1j*np.einsum('ijk,ik,ik,j->ij', theta_ijk, arrGijtilde, sig_z_sig_p, np.conj(sig_p))
    tempsum += dict_corrs_10['sig_p.sig_m']*1j*np.einsum('ijk,ik,kj,i->ij', theta_ijk, arrGijtilde, sig_p_sig_m, sig_z)
    tempsum += dict_corrs_10['sig_z.sig_p']*1j*np.einsum('ijk,ik,ij,k->ij', theta_ijk, arrGijtilde, np.conj(sig_z_sig_p), sig_p)
    tempsum += (1-dict_corrs_10['sig_z.sig_p'])*temp2 + (1-dict_corrs_10['sig_p.sig_m'])*temp2  + (1-dict_corrs_10['sig_z.sig_p'])*temp2 
    tempsum += -2*temp2


    return tempsum




##########################################################################################

#final EOM based on which terms are two point and which are one point correlators

def f_sig_dot_vec(t, sig_list, drive=1, detuning=-detuning_list[0]):
    
    sig_z = sig_list[:Natoms] 
    sig_p = sig_list[Natoms:int(2*Natoms)] 
    
    sig_z_sig_z = np.reshape(sig_list[int(2*Natoms):int(2*Natoms+Natoms**2)], (Natoms, Natoms))
    sig_z_sig_p = np.reshape(sig_list[int(2*Natoms+Natoms**2):int(2*Natoms+2*(Natoms**2))], (Natoms, Natoms))
    sig_p_sig_p = np.reshape(sig_list[int(2*Natoms+2*(Natoms**2)):int(2*Natoms+3*(Natoms**2))], (Natoms, Natoms))
    sig_p_sig_m = np.reshape(sig_list[int(2*Natoms+3*(Natoms**2)):], (Natoms, Natoms))
    
    sig_z_dot = (f_sig_z_dot(sig_z, sig_p, sig_p_sig_m, drive)) #.flatten()
    sig_p_dot = (f_sig_p_dot(sig_z, sig_p, sig_z_sig_p, detuning, drive)) #.flatten()

    sig_z_sig_z_dot = dict_corrs_10['sig_z.sig_z']*(f_sig_z_sig_z_dot(sig_z, sig_p, sig_z_sig_z, sig_z_sig_p, sig_p_sig_m, detuning, drive)).flatten()
    sig_z_sig_p_dot = dict_corrs_10['sig_z.sig_p']*(f_sig_z_sig_p_dot(sig_z, sig_p, sig_z_sig_z, sig_z_sig_p, sig_p_sig_m, sig_p_sig_p, detuning, drive)).flatten()
    sig_p_sig_p_dot = dict_corrs_10['sig_p.sig_p']*(f_sig_p_sig_p_dot(sig_z, sig_p, sig_z_sig_p, sig_p_sig_p, detuning, drive)).flatten()
    sig_p_sig_m_dot = dict_corrs_10['sig_p.sig_m']*(f_sig_p_sig_m_dot(sig_z, sig_p, sig_z_sig_z, sig_z_sig_p, sig_p_sig_m, sig_p_sig_p, detuning, drive)).flatten()

    sig_dot_mat = np.concatenate((sig_z_dot,sig_p_dot, sig_z_sig_z_dot, sig_z_sig_p_dot, sig_p_sig_p_dot, sig_p_sig_m_dot))
    return sig_dot_mat.flatten()

'''
def f_reached_equilibrium(t, y, drive, detuning):
    return np.linalg.norm(f_sig_dot_vec(t, y, drive, detuning)) - tol_params_equil

f_reached_equilibrium.terminal = True
f_reached_equilibrium.direction = -1
'''

##########################################################################
##########################################################################

num_single_particle_ops = int(Natoms*2) # sig_z, sig_p

# no. of two pt ops

total_num_double = int(4*(Natoms**2)) #*N**2 for total number for all atoms

'''
# more functions

def ketEm(me):
    temp = np.zeros(HSsize)
    temp[int(me + adde)] = 1
    return temp


def ketGn(mg):
    temp = np.zeros(HSsize)
    temp[int(mg + addg + 2*fe + 1)] = 1
    return temp


def sigma_emgn(me, mg):
    return np.outer(ketEm(me), ketGn(mg))

def sigma_emem(me1, me2):
    return np.outer(ketEm(me1), ketEm(me2))

def sigma_gngn(mg1, mg2):
    return np.outer(ketGn(mg1), ketGn(mg2))

def sigma_gnem(mg, me):
    return np.outer(ketGn(mg), ketEm(me))

def sparse_trace(A):
    return A.diagonal().sum()


def funcOp(kinput, A):
    k = kinput+1
    if (k> Natoms or k<1):
        return "error"
    elif k == 1:
        temp = csr_matrix(A)
        for i in range(1, Natoms):
            temp = csr_matrix(sparse.kron(temp, np.identity(HSsize)))
        return temp
    else:
        temp = csr_matrix(np.identity(HSsize))
        for i in range(2, k):
            temp = csr_matrix(sparse.kron(temp, np.identity(HSsize)))
        temp = csr_matrix(sparse.kron(temp, A))
        for i in range(k, Natoms):
            temp = csr_matrix(sparse.kron(temp, np.identity(HSsize)))
        return temp


def f_one_pt_fns(rho_sol_dr):
    single_particle_ops_pop = np.zeros(num_single_particle_ops, complex)
    index = 0
    for k in range(0, Natoms):
        for n1 in range(0, deg_g):
            for n2 in range(0, deg_g):
                single_particle_ops_pop[index] = sparse_trace(sparse.csr_matrix.dot(rho_sol_dr,funcOp(k, sigma_gngn(n1-fg, n2-fg))))
                index += 1
        for m1 in range(0, deg_e):
            for m2 in range(0, deg_e):
                single_particle_ops_pop[index] = sparse_trace(sparse.csr_matrix.dot(rho_sol_dr,funcOp(k, sigma_emem(m1-fe, m2-fe))))
                index += 1
        for m1 in range(0, deg_e):
            for n2 in range(0, deg_g):
                single_particle_ops_pop[index] = sparse_trace(sparse.csr_matrix.dot(rho_sol_dr,funcOp(k, sigma_emgn(m1-fe, n2-fg))))
                index += 1
    return single_particle_ops_pop

#correlations of the form sig_ab^k . sig_cd^l 

def f_two_pt_corrs(rho_sol_dr):
    two_particle_corrs = np.zeros(total_num_double*(Natoms**2), complex)
    index = 0
    for n1 in range(0, Natoms):
        for n2 in range(0, Natoms):
            for iia in range(0, deg_g):
                a = iia-fg
                for iib in range(0, deg_g):
                    b = iib-fg
                    for iic in range(0, deg_g):
                        c = iic-fg
                        for iid in range(0, deg_g):
                            d = iid-fg
                            if n1!=n2:
                                two_particle_corrs[index] = sparse_trace(sparse.csr_matrix.dot(sparse.csr_matrix.dot(rho_sol_dr,funcOp(n1, sigma_gngn(a, b))),funcOp(n2, sigma_gngn(c, d)))) 
                            index += 1
            for iia in range(0, deg_e):
                a = iia-fe
                for iib in range(0, deg_g):
                    b = iib-fg
                    for iic in range(0, deg_g):
                        c = iic-fg
                        for iid in range(0, deg_g):
                            d = iid-fg
                            if n1!=n2:
                                two_particle_corrs[index] = sparse_trace(sparse.csr_matrix.dot(sparse.csr_matrix.dot(rho_sol_dr,funcOp(n1, sigma_emgn(a, b))),funcOp(n2, sigma_gngn(c, d)))) 
                            index += 1
            for iia in range(0, deg_e):
                a = iia-fe
                for iib in range(0, deg_g):
                    b = iib-fg
                    for iic in range(0, deg_g):
                        c = iic-fg
                        for iid in range(0, deg_e):
                            d = iid-fe
                            if n1!=n2:
                                two_particle_corrs[index] = sparse_trace(sparse.csr_matrix.dot(sparse.csr_matrix.dot(rho_sol_dr,funcOp(n1, sigma_emgn(a, b))),funcOp(n2, sigma_gnem(c, d)))) 
                            index += 1
            for iia in range(0, deg_e):
                a = iia-fe
                for iib in range(0, deg_e):
                    b = iib-fe
                    for iic in range(0, deg_g):
                        c = iic-fg
                        for iid in range(0, deg_g):
                            d = iid-fg
                            if n1!=n2:
                                two_particle_corrs[index] = sparse_trace(sparse.csr_matrix.dot(sparse.csr_matrix.dot(rho_sol_dr,funcOp(n1, sigma_emem(a, b))),funcOp(n2, sigma_gngn(c, d)))) 
                            index += 1
            for iia in range(0, deg_e):
                a = iia-fe
                for iib in range(0, deg_g):
                    b = iib-fg
                    for iic in range(0, deg_e):
                        c = iic-fe
                        for iid in range(0, deg_g):
                            d = iid-fg
                            if n1!=n2:
                                two_particle_corrs[index] = sparse_trace(sparse.csr_matrix.dot(sparse.csr_matrix.dot(rho_sol_dr,funcOp(n1, sigma_emgn(a, b))),funcOp(n2, sigma_emgn(c, d)))) 
                            index += 1
            for iia in range(0, deg_e):
                a = iia-fe
                for iib in range(0, deg_e):
                    b = iib-fe
                    for iic in range(0, deg_e):
                        c = iic-fe
                        for iid in range(0, deg_g):
                            d = iid-fg
                            if n1!=n2:
                                two_particle_corrs[index] = sparse_trace(sparse.csr_matrix.dot(sparse.csr_matrix.dot(rho_sol_dr,funcOp(n1, sigma_emem(a, b))),funcOp(n2, sigma_emgn(c, d)))) 
                            index += 1
            for iia in range(0, deg_e):
                a = iia-fe
                for iib in range(0, deg_e):
                    b = iib-fe
                    for iic in range(0, deg_e):
                        c = iic-fe
                        for iid in range(0, deg_e):
                            d = iid-fe
                            if n1!=n2:
                                two_particle_corrs[index] = sparse_trace(sparse.csr_matrix.dot(sparse.csr_matrix.dot(rho_sol_dr,funcOp(n1, sigma_emem(a, b))),funcOp(n2, sigma_emem(c, d)))) 
                            index += 1

    return two_particle_corrs
'''
###################################################################################

# initial condition

total_num = total_num_double+num_single_particle_ops

#initialising system in which all atoms are in the ground state    

initial_sig_vec = np.zeros((total_num),complex)
for n1 in range(0, Natoms):
    initial_sig_vec[dict_ops['z',n1]] = -1.0 + 0*1j # sigma_z = -0.5 for each atom
    for n2 in range(0, Natoms):
        if n1!=n2:
            initial_sig_vec[dict_ops['z,z',n1,n2]] = 1.0 + 0*1j
                    
    
'''
psi_in = (np.array([0,0,1,0])+np.array([0,0,0,1]))/np.sqrt(2)
psi_in_all = psi_in + 0*1j
for i in range(1, Natoms):
    psi_in_all = np.kron(psi_in_all, psi_in)
rho_in = np.outer(psi_in_all, np.conj(psi_in_all))  
rho_in = csr_matrix(rho_in)
single_ops = f_one_pt_fns(rho_in)
two_ops = f_two_pt_corrs(rho_in)
initial_sig_vec = np.zeros((total_num),complex)
initial_sig_vec[:num_single_particle_ops] = single_ops
initial_sig_vec[num_single_particle_ops:] = two_ops      
'''
    
initial_sig_vec = initial_sig_vec +0*1j

########################################################################

print('Run started at:'+str(time.time()), flush=True)

#driven evolution from a ground state to get to the steady state

ta1 = time.time()
sol = solve_ivp(f_sig_dot_vec, t_range_dr, initial_sig_vec, method='RK45', t_eval=t_vals_dr, dense_output=False, events=None, atol = 1e-20, rtol = 1e-18)#, args=[1, -detuning_list[i_det]])  'RK45'
tb1 = time.time()
runtime1 = tb1-ta1

print('runtime cumulant equilibrate = '+str(runtime1), flush=True)

phase_list_single = np.exp(1j*np.einsum('nx,x->n', rvecall, kvec))


total_exc_dr = np.zeros(len(t_vals_dr))
total_Sx_dr = np.zeros(len(t_vals_dr))
total_Sy_dr = np.zeros(len(t_vals_dr))
total_Sp_with_phase_dr = np.zeros(len(t_vals_dr), complex)
forward_intensity_dr = np.zeros(len(t_vals_dr), complex)
sig_list_dr = np.zeros((len(t_vals_dr), total_num), complex)

for t in range(0, len(sol.t)): #len(t_vals_dr)):
    index = 0
    sig_list_tmp = sol.y[:,t]
    sig_z = sig_list_tmp[:Natoms] 
    sig_p = sig_list_tmp[Natoms:int(2*Natoms)] 
    sig_p_sig_m = np.reshape(sig_list_tmp[int(2*Natoms+3*(Natoms**2)):], (Natoms, Natoms))

    total_exc_dr[t] = 0.5*np.sum(sig_z).real
    temp_total_Sp_dr = np.sum(sig_p)
    total_Sx_dr[t] = (temp_total_Sp_dr + np.conj(temp_total_Sp_dr)).real
    total_Sy_dr[t] = (-1j*(temp_total_Sp_dr - np.conj(temp_total_Sp_dr))).real
    forward_intensity_dr[t] = np.einsum('kj,kj,kj->', delta_ij, phase_array, sig_p_sig_m)
    total_Sp_with_phase_dr[t] = np.einsum('k,k->', sig_p, phase_list_single)
    
    sig_list_dr[t] = sig_list_tmp

total_exc_dr += 0.5*Natoms
forward_intensity_dr += total_exc_dr

# save data

save_exc = np.zeros((len(t_vals_dr), 2))
save_exc[:, 0] = t_vals_dr
save_exc[:, 1] = total_exc_dr


hf = h5py.File(direc+'Data_Cumulant_dynamics_to_equil_'+h5_title_dr, 'w')
hf.create_dataset('total_exc', data=save_exc, compression="gzip", compression_opts=9)
hf.create_dataset('total_Sx', data=total_Sx_dr, compression="gzip", compression_opts=9)
hf.create_dataset('total_Sy', data=total_Sy_dr, compression="gzip", compression_opts=9)
hf.create_dataset('forward_intensity_dr', data=forward_intensity_dr, compression="gzip", compression_opts=9)
hf.create_dataset('rvecall', data=rvecall, compression="gzip", compression_opts=9)
hf.create_dataset('sig_list_dr', data=sig_list_dr, compression="gzip", compression_opts=9)
hf.create_dataset('total_Sp_with_phase_dr', data=total_Sp_with_phase_dr, compression="gzip", compression_opts=9)
hf.close()

tc1 = time.time()

print('runtime data analysis and saving = '+str(tc1-tb1), flush=True)

print("All runs done. May all your codes run this well! No decay run. :)", flush=True)

##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
