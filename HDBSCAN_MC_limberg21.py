import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
import matplotlib.cm as cm
from sklearn import preprocessing
import pylab as pl
import os
from matplotlib import pyplot
import pandas as pd
#%matplotlib inline

from astropy.stats import biweight_scale
from astropy.stats import biweight_location

import random
random.seed(999)

from random import shuffle
from random import randrange
from random import  uniform

#------------------------------------------------------------
# Load the dataset
from astropy.table import Table
data =Table.read("HKHES+BNB_GaiaEDR3_LowResRV_NoDisk.csv") 
#disk =Table.read("your_disk_sample.csv") 

### VERY IMPORTANT: THE INPUT DATA MUST BE NORMALIZED FOR THE CLUSTERING TO WORK PROPERLY

#disk = disk.filled()
data = data.filled()

# data for clustering
energy = data["energy50"]
jr = data["j_r50"]
jphi = data["j_phi50"]
jz = data["j_z50"]

#star_id = data["star_id"]

# Data for saving later
ecc = data["ecc50"]
vphi = data["v_phi50"]
vr = data["v_r50"]
vz = data["v_z50"]
vy = data["vy50"]
vx = data["vx50"]
#cfe = data["[C/Fe]"]
Vmag = data["Vmag_GaiaEDR3"]
ra = data["RA_GaiaEDR3"]
dec = data["DEC_GaiaEDR3"]
dist = data["Dist_GaiaEDR3_corr"]
Vr = data["RV_LowRes"]
feh = data["FeH"]
Teff = data["Teff"]
logg = data["logg"]
x = data["xgal50"]
y = data["ygal50"]
z = data["zgal50"]
names = data["Gaia EDR3 ID"]
incl = data["incl50"]
zmax = data["zmax50"]
rapo = data["rapo50"]
peri = data["rperi50"]
Lx = data["Lx50"]
Ly = data["Ly50"]
Lz = data["Lz50"]


# IMPORTANT: here I am normalizing via biweight scale/loocation, but in LIMBERG+2021 we used the preprocessing task from sklearn 

# stack the results for computation
X_init = np.vstack([energy, jr, jphi, jz]).T 

loc_energy = biweight_location(energy)
scale_energy = biweight_scale(energy)

loc_jr = biweight_location(jr)
scale_jr = biweight_scale(jr)

loc_jphi = biweight_location(jphi)
scale_jphi = biweight_scale(jphi)

loc_jz = biweight_location(jz)
scale_jz = biweight_scale(jz)

norm_energy = []
norm_jr     = []
norm_jphi   = []
norm_jz     = []

for i in range(len(names)):
    norm_energy.append( (energy[i] - loc_energy)/scale_energy )
    norm_jr.append( (jr[i] - loc_jr)/scale_jr )
    norm_jphi.append( (jphi[i] - loc_jphi)/scale_jphi )
    norm_jz.append( (jz[i] - loc_jz)/scale_jz )

X = np.vstack([norm_energy, norm_jr, norm_jphi, norm_jz]).T

#scale data
#X = preprocessing.scale(X_init)

energy_std = data["energy_std"]
jphi_std = data["jphi_std"]
jr_std = data["jr_std"]
jz_std = data["jz_std"]

N_MC = 1000 #1000

MC_E = []
MC_jr = []
MC_jphi = []
MC_jz = []

for i in np.arange(0, len(names)):
    for j in np.arange(0, N_MC):
        MC_E.append(energy[i] + np.random.randn() * energy_std[i] )
        MC_jr.append(jr[i] + np.random.randn() * jr_std[i] )
        MC_jphi.append(jphi[i] + np.random.randn() * jphi_std[i] )
        MC_jz.append(jz[i] + np.random.randn() * jz_std[i] )


loc_MC_E = biweight_location(MC_E)
scale_MC_E = biweight_scale(MC_E)

loc_MC_jr = biweight_location(MC_jr)
scale_MC_jr = biweight_scale(MC_jr)

loc_MC_jphi = biweight_location(MC_jphi)
scale_MC_jphi = biweight_scale(MC_jphi)

loc_MC_jz = biweight_location(MC_jz)
scale_MC_jz = biweight_scale(MC_jz)

norm_MC_E = []
norm_MC_jr     = []
norm_MC_jphi   = []
norm_MC_jz     = []

for i in range(len(names)*N_MC):
    norm_MC_E.append( (MC_E[i] - loc_MC_E)/scale_MC_E )
    norm_MC_jr.append( (MC_jr[i] - loc_MC_jr)/scale_MC_jr )
    norm_MC_jphi.append( (MC_jphi[i] - loc_MC_jphi)/scale_MC_jphi )
    norm_MC_jz.append( (MC_jz[i] - loc_MC_jz)/scale_MC_jz )

test_X = np.vstack([norm_MC_E, norm_MC_jr, norm_MC_jphi, norm_MC_jz]).T


X_init = np.vstack([energy, ecc, jphi, jz, jr, vphi, vz, vr, vx, vy, logg,incl, dist, Vr, feh, Teff, x, y, z, zmax, rapo, peri, Lx, Ly, Lz,names,ra,dec,Vmag]).T

"""
#------------------------------------------------------------
# Load external sample
test = Table.read("your_external_sample.csv")

### VERY IMPORTANT: THE INPUT DATA MUST BE SCALED FOR THE CLUSTERING TO WORK PROPERLY

test = test.filled()

test_star_id = test["star_id"]
test_energy = test["energy50"]
test_jr = test["j_r50"]
test_jphi = test["j_phi50"]
test_jz = test["j_z50"]

test_energy16 = test["energy16"]
test_energy84 = test["energy84"]

test_jr16 = test["j_r16"]
test_jr84 = test["j_r84"]

test_jphi16 = test["j_phi16"]
test_jphi84 = test["j_phi84"]

test_jz16 = test["j_z16"]
test_jz84 = test["j_z84"]

N_MC = 1 #1000

MC_E = []
MC_jr = []
MC_jphi = []
MC_jz = []
#2.
#2. 
for i in np.arange(0, len(test_star_id)):
    for j in np.arange(0, N_MC/2):
        MC_E.append(test_energy[i] + 2. * (test_energy[i] - test_energy16[i]) )
        MC_E.append(test_energy[i] + 2. * (test_energy84[i] - test_energy[i]) )
        MC_jr.append(test_jr[i] + 2. * (test_jr[i] - test_jr16[i]) )
        MC_jr.append(test_jr[i] + 2. * (test_jr84[i] - test_jr[i]) )
        MC_jphi.append(test_jphi[i] + 2. * (test_jphi[i] - test_jphi16[i]) )
        MC_jphi.append(test_jphi[i] + 2. * (test_jphi84[i] - test_jphi[i]) )
        MC_jz.append(test_jz[i] + 2. * (test_jz[i] - test_jz16[i]) )
        MC_jz.append(test_jz[i] + 2. * (test_jz84[i] - test_jz[i]) )

#np.random.randn()
# stack the results for computation
Y_init = np.vstack([MC_E, MC_jr, MC_jphi, MC_jz]).T

#SCALE data
Y = preprocessing.scale(Y_init)

Y_init = np.vstack([test_energy, test_jphi, test_jz, test_jr, test_star_id]).T
"""

def connectpoints(x,y,p1,p2):
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    plt.plot([x1,x2],[y1,y2],'k', lw=2, zorder=-2)

def plot_clusters(data, test1, test4 ,data_init, test_init, algorithm, j, k, args, kwds):
    method = 'leaf'
    epsilon = 0.
    start_time = time.time()
    clusterer = hdbscan.HDBSCAN(min_cluster_size=int(j), min_samples=int(k), prediction_data=True, cluster_selection_method=method, cluster_selection_epsilon=epsilon)
    clusterer.fit(X)
    labels = clusterer.labels_

    occur = 0
    for i in np.arange(max(labels)+1):
        occur = occur + list(labels).count(i)
    avg = occur/(max(labels)+1)
    avg = round(avg,2)


    # these next few lines are responsible for taking the generated monte carlo realizations and throw them back into the 'nominal' clusters
    test_labels, strength = hdbscan.approximate_predict(clusterer, test_X)

    print_test = []
    aux_vec = []
    for i in np.arange(0,len(test_labels)):
        aux_vec.append(test_labels[i])
        if len(aux_vec) == 1000:
            aux_vec.sort() 
            print_test.append(aux_vec)
            aux_vec = []
    ############################################################################################

    probs =  clusterer.probabilities_
    out   =  clusterer.outlier_scores_
    end_time = time.time()
    #print(end_time - start_time)
    N_cluster = max(labels)+1
    N_stars = len(names)
    print(N_cluster, "clusters found")
    print("min_samples: ", k)
    fig = plt.figure(figsize=(17, 17))
    plt.rcParams.update({'font.size': 10})

    
# Define a list of colors
    colors2= ["lime", "orange", "purple", "goldenrod","black", "deepskyblue", "springgreen", "chocolate", "cornflowerblue", "orchid", "red", "royalblue", "darkgreen", "saddlebrown", "gold", "darkcyan", "firebrick", "seagreen", "darkviolet", "lightcoral", "mediumpurple", "dimgrey"]
    colors2 = 10 * colors2
    shuffle(colors2)

    marker = ['d', '^', 'o', 'P', 'h', '<', 's', 'X', 'p', 'v', '>']
    marker = 15 * marker
    shuffle(marker)

### Create things necessary for plotting the legend
    noise = mlines.Line2D([],[], color='silver', alpha=0.5, marker='o', linestyle='None', markersize=2.0, label="BG Star")
    handles = [noise]
    for i in np.arange(max(labels)+1):
        handles.append(mlines.Line2D([],[], color=colors2[i], alpha=1.0, marker=marker[i], linestyle='None', markersize=7, label="Cluster "+str(i+1), markeredgecolor='k', mew=0.5))

########### Plot1 ###########
    ax1 = fig.add_subplot(4,4,1)

    ax1.set_axisbelow(True)
    plt.grid(b=True, lw=0.5, zorder=-2)

    #plt.scatter(-disk["j_phi50"]/1000, disk["energy50"]/100000, c='steelblue' , zorder=-5, s=5, alpha=0.1, marker='o')

    plt.scatter(-np.array(data_init.T[2][(labels==-1)], dtype=np.float32)/1000, np.array(data_init.T[0][(labels==-1)], dtype=np.float32)/100000, c='silver', s=5, alpha=0.3, marker='o')

    ## plots the data that has been assigned to a group (color coded) ##
    for ii in np.arange(max(labels)+1):
        plt.scatter(-np.array(data_init.T[2][(labels==ii)], dtype=np.float32)/1000, np.array(data_init.T[0][(labels==ii)], dtype=np.float32)/100000, c=colors2[ii], s=30, alpha=1, marker=marker[ii], edgecolor='k', lw=0.6)


    plt.xlim(-4,3)    
    plt.ylim(-2.0, -0.8)
    plt.xlabel(r'$J_{\phi}$ ($\times 10^3$ km s$^{-1}$ kpc)', fontsize=15)
    plt.ylabel(r'E ($\times 10^5$ km$^2$ s$^{-2}$)', fontsize=15)


########### Plot2 ###########
    ax1 = fig.add_subplot(4,4,2) 

    ax1.set_axisbelow(True)
    plt.grid(b=True, lw=0.5, zorder=-2)

    plt.scatter(-np.array(data_init.T[2][(labels==-1)], dtype=np.float32)/1000, np.array(data_init.T[3][(labels==-1)], dtype=np.float32)/1000, c='silver', s=5, alpha=0.3, marker='o')
        
    for ii in np.arange(max(labels)+1):
        plt.scatter(-np.array(data_init.T[2][(labels==ii)], dtype=np.float32)/1000, np.array(data_init.T[3][(labels==ii)], dtype=np.float32)/1000, c=colors2[ii], s=30, alpha=1, marker=marker[ii], edgecolor='k', linewidths=0.6)   

    plt.xlim(-4, 3.0)  
    plt.ylim(0 , 3.0)
    plt.xlabel(r'$J_{\phi}$ ($\times 10^3$ km s$^{-1}$ kpc)', fontsize=15)
    plt.ylabel(r'$J_z$ ($\times 10^3$ km s$^{-1}$ kpc)', fontsize=15)



########### Plot3 ###########
    ax1 = fig.add_subplot(4,4,3)

    ax1.set_axisbelow(True)
    plt.grid(b=True, lw=0.5, zorder=-2)

    plt.scatter(np.array(data_init.T[4][(labels==-1)], dtype=np.float32)/1000, np.array(data_init.T[3][(labels==-1)], dtype=np.float32)/1000, c='silver', s=5, alpha=0.3, marker='o')

    for ii in np.arange(max(labels)+1):

        plt.scatter(np.array(data_init.T[4][(labels==ii)], dtype=np.float32)/1000, np.array(data_init.T[3][(labels==ii)], dtype=np.float32)/1000, c=colors2[ii], s=30, alpha=1, marker=marker[ii], edgecolor='k', linewidths=0.6)  

    plt.xlim(0,2.5)    
    plt.ylim(0,2.5)    
    plt.xlabel(r'$J_R$ ($\times 10^3$ km s$^{-1}$ kpc)', fontsize=15)
    plt.ylabel(r'$J_z$ ($\times 10^3$ km s$^{-1}$ kpc)', fontsize=15)

    plt.text(2.4, 2.35, '%0.0f total stars' % (N_stars), fontsize=10, ha='right')
    plt.text(2.4, 2.20, '%0.0f clusters found' % (N_cluster), fontsize=10, ha='right')
    plt.text(2.4, 2.05, 'min_cluster_size: %0.0f' % (j), fontsize=10, ha='right')
    plt.text(2.4, 1.90, 'min_samples: %0.0f' % (k), fontsize=10, ha='right')
    plt.text(2.4, 1.75, 'method = '+str(method), fontsize=10, ha='right')
    plt.text(2.4, 1.60, 'Avg. members = '+str(avg), fontsize=10, ha='right')
    plt.text(2.4, 1.45, 'Epsilon = '+str(epsilon), fontsize=10, ha='right')

########### Plot3 ###########
    ax1 = fig.add_subplot(4,4,4)

    ax1.set_axisbelow(True)
    plt.grid(b=True, lw=0.5, zorder=-2)

    plt.scatter(np.array(data_init.T[20][(labels==-1)], dtype=np.float32), np.array(data_init.T[19][(labels==-1)], dtype=np.float32), c='silver', s=5, alpha=0.3, marker='o', zorder=-2)

    for ii in np.arange(max(labels)+1):

        plt.scatter(np.array(data_init.T[20][(labels==ii)], dtype=np.float32), np.array(data_init.T[19][(labels==ii)], dtype=np.float32), c=colors2[ii], s=30, alpha=1, marker=marker[ii], edgecolor='k', linewidths=0.6)  

    plt.xlim(0, 30)    
    plt.ylim(0, 30)    
    plt.xlabel(r'$R_{max}$ (kpc)', fontsize=15)
    plt.ylabel(r'$Z_{max}$ (kpc)', fontsize=15)

    plt.legend(handles=handles[0:999], facecolor='white', frameon=False, fontsize=10, loc='upper right', bbox_to_anchor=(1.55,1.0))

    plt.hlines(3, 0, 100, colors='k', alpha=0.8, linestyles='dashed', linewidths=2, zorder=-1)

    #plt.scatter(disk["rapo50"], disk["zmax50"], c='steelblue' , zorder=-5, s=5, alpha=0.1, marker='o')

########### Plot3 ###########
    ax1 = fig.add_subplot(4,4,5)

    ax1.set_axisbelow(True)
    plt.grid(b=True, lw=0.5, zorder=-2)

    plt.scatter(-np.array(data_init.T[5][(labels==-1)], dtype=np.float32), np.array(data_init.T[7][(labels==-1)], dtype=np.float32), c='silver', s=5, alpha=0.3, marker='o')

    for ii in np.arange(max(labels)+1):
        plt.scatter(-np.array(data_init.T[5][(labels==ii)], dtype=np.float32), np.array(data_init.T[7][(labels==ii)], dtype=np.float32), c=colors2[ii], s=30, alpha=1, marker=marker[ii], edgecolor='k', linewidths=0.6)  

    plt.xlim(-400, 300)    
    plt.ylim(-400, 400)    
    plt.xlabel(r'$v_{\phi}$ (km s$^{-1}$)', fontsize=15)
    plt.ylabel(r'$v_R$ (km s$^{-1}$)', fontsize=15)


########### Plot3 ###########
    ax1 = fig.add_subplot(4,4,6)

    ax1.set_axisbelow(True)
    plt.grid(b=True, lw=0.5, zorder=-2)

    plt.scatter(-np.array(data_init.T[5][(labels==-1)], dtype=np.float32), np.array(data_init.T[6][(labels==-1)], dtype=np.float32), c='silver', s=5, alpha=0.3, marker='o')

    for ii in np.arange(max(labels)+1):
        plt.scatter(-np.array(data_init.T[5][(labels==ii)], dtype=np.float32), np.array(data_init.T[6][(labels==ii)], dtype=np.float32), c=colors2[ii], s=30, alpha=1, marker=marker[ii], edgecolor='k', linewidths=0.6)  

    plt.xlim(-400, 300)    
    plt.ylim(-400, 400)    
    plt.xlabel(r'$v_{\phi}$ (km s$^{-1}$)', fontsize=15)
    plt.ylabel(r'$v_z$ (km s$^{-1}$)', fontsize=15)


########### Plot3 ###########
    ax1 = fig.add_subplot(4,4,7)

    ax1.set_axisbelow(True)
    plt.grid(b=True, lw=0.5, zorder=-2)

    plt.scatter(-np.array(data_init.T[7][(labels==-1)], dtype=np.float32), np.array(data_init.T[6][(labels==-1)], dtype=np.float32), c='silver', s=5, alpha=0.3, marker='o')

    for ii in np.arange(max(labels)+1):
        plt.scatter(-np.array(data_init.T[7][(labels==ii)], dtype=np.float32), np.array(data_init.T[6][(labels==ii)], dtype=np.float32), c=colors2[ii], s=30, alpha=1, marker=marker[ii], edgecolor='k', linewidths=0.6)  

    plt.xlim(-400, 400)    
    plt.ylim(-400, 400)    
    plt.xlabel(r'$v_R$ (km s$^{-1}$)', fontsize=15)
    plt.ylabel(r'$v_z$ (km s$^{-1}$)', fontsize=15)


########### Plot8 sqrt( v_r^2 + v_z^2 ) x v_phi ###########
    ax1 = fig.add_subplot(4,4,8)

    ax1.set_axisbelow(True)
    plt.grid(b=True, lw=0.5, zorder=-2)

    plt.scatter(-np.array(data_init.T[5][(labels==-1)], dtype=np.float32), np.sqrt( np.array(data_init.T[6][(labels==-1)], dtype=np.float32)**2 + np.array(data_init.T[7][(labels==-1)], dtype=np.float32)**2), c='silver', s=5, alpha=0.3, marker='o', zorder=-1)

    ## plots the data that has been assigned to a group (color coded) ##
    for ii in np.arange(max(labels)+1):

        plt.scatter( -np.array(data_init.T[5][(labels==ii)], dtype=np.float32), np.sqrt( np.array(data_init.T[6][(labels==ii)], dtype=np.float32)**2 + np.array(data_init.T[7][(labels==ii)], dtype=np.float32)**2 ), c=colors2[ii], s=30, alpha=1, marker=marker[ii], edgecolor='k', linewidths=0.6)

    plt.xlim(-400, 300)    
    plt.ylim(0, 400)
    plt.xlabel(r'$v_{\phi}$ (km s$^{-1}$)', fontsize=15)
    plt.ylabel(r'$\sqrt{ v_R^2 + v_z^2 }$ (km s$^{-1}$)', fontsize=15)

    circle = plt.Circle((243.0,0), 220, edgecolor='k', alpha=0.8, lw=3, fill=False, ls='dashed', zorder=-1, facecolor='white')
    ax1.add_artist(circle)

    #plt.scatter(-disk["v_phi50"], np.sqrt(disk["v_r50"]**2 + disk["v_z50"]**2), c='steelblue' , zorder=-5, s=5, alpha=0.1, marker='o')

########### Plot3 ###########
    ax1 = fig.add_subplot(4,4,9)

    ax1.set_axisbelow(True)
    plt.grid(b=True, lw=0.5, zorder=-2)

    plt.scatter(np.array(data_init.T[16][(labels==-1)], dtype=np.float32), np.array(data_init.T[17][(labels==-1)], dtype=np.float32), c='silver', s=5, alpha=0.3, marker='o', zorder=-2)

    for ii in np.arange(max(labels)+1):
        plt.scatter(np.array(data_init.T[16][(labels==ii)], dtype=np.float32), np.array(data_init.T[17][(labels==ii)], dtype=np.float32), c=colors2[ii], s=30, alpha=1, marker=marker[ii], edgecolor='k', linewidths=0.6)  

    plt.xlim(-15, 5)    
    plt.ylim(-10, 10)    
    plt.xlabel(r'$X_{Gal}$ (kpc)', fontsize=15)
    plt.ylabel(r'$Y_{Gal}$ (kpc)', fontsize=15)

    circle = plt.Circle((0,0),3, edgecolor='k', alpha=0.8, lw=2, fill=False, ls='dashed', zorder=-1, facecolor='white')
    ax1.add_artist(circle)


########### Plot3 ###########
    ax1 = fig.add_subplot(4,4,10)

    ax1.set_axisbelow(True)
    plt.grid(b=True, lw=0.5, zorder=-2)

    plt.scatter(np.array(data_init.T[16][(labels==-1)], dtype=np.float32), np.array(data_init.T[18][(labels==-1)], dtype=np.float32), c='silver', s=5, alpha=0.3, marker='o', zorder=-2)

    for ii in np.arange(max(labels)+1):
        plt.scatter(np.array(data_init.T[16][(labels==ii)], dtype=np.float32), np.array(data_init.T[18][(labels==ii)], dtype=np.float32), c=colors2[ii], s=30, alpha=1, marker=marker[ii], edgecolor='k', linewidths=0.6)  

    plt.xlim(-15, 5)    
    plt.ylim(-10, 10)    
    plt.xlabel(r'$X_{Gal}$ (kpc)', fontsize=15)
    plt.ylabel(r'$Z_{Gal}$ (kpc)', fontsize=15)

    circle = plt.Circle((0,0),3, edgecolor='k', alpha=0.8, lw=2, fill=False, ls='dashed', zorder=-5, facecolor='white')
    ax1.add_artist(circle)

    #plt.scatter(disk["xgal50"], disk["zgal50"], c='steelblue' , zorder=-1, s=5, alpha=0.1, marker='o')

########### Plot3 ###########
    ax1 = fig.add_subplot(4,4,11)

    ax1.set_axisbelow(True)
    plt.grid(b=True, lw=0.5, zorder=-2)

    plt.scatter(np.array(data_init.T[17][(labels!=1)], dtype=np.float32), np.array(data_init.T[18][(labels!=1)], dtype=np.float32), c='silver', s=5, alpha=0.3, marker='o')

    for ii in np.arange(max(labels)+1):
        plt.scatter(np.array(data_init.T[17][(labels==4)], dtype=np.float32), np.array(data_init.T[18][(labels==4)], dtype=np.float32), c=colors2[4], s=50, alpha=1, marker=marker[4], edgecolor='k', linewidths=0.6)
        plt.quiver(np.array(data_init.T[17][(labels==4)], dtype=np.float32),np.array(data_init.T[18][(labels==4)], dtype=np.float32), np.array(-data_init.T[9][(labels==4)], dtype=np.float32), np.array(data_init.T[6][(labels==4)], dtype=np.float32), color=colors2[4], scale=2000, edgecolor='black', linewidth = 0.5)

    for ii in np.arange(max(labels)+1):
        plt.scatter(np.array(data_init.T[17][(labels==3)], dtype=np.float32), np.array(data_init.T[18][(labels==3)], dtype=np.float32), c=colors2[3], s=50, alpha=1, marker=marker[3], edgecolor='k', linewidths=0.6)
        plt.quiver(np.array(data_init.T[17][(labels==3)], dtype=np.float32),np.array(data_init.T[18][(labels==3)], dtype=np.float32), np.array(-data_init.T[9][(labels==3)], dtype=np.float32), np.array(data_init.T[6][(labels==3)], dtype=np.float32), color=colors2[3], scale=2000, edgecolor='black', linewidth = 0.5)

    plt.xlim( 5, -5)    
    plt.ylim(-5,  5)    
    plt.xlabel(r'$Y_{Gal}$ (kpc)', fontsize=15)
    plt.ylabel(r'$Z_{Gal}$ (kpc)', fontsize=15)


########### Plot9 Act space map ###########
    ax1 = fig.add_subplot(4,4,12)

    ax1.set_axisbelow(True)
    plt.grid(b=True, lw=0.5, zorder=-2)


    plt.text(-1.15, 0, r'Retrograde', fontsize=10, color='k', fontweight='normal', rotation=90, ha='center', va='center')
    plt.text(1.15, 0, r'Prograde', fontsize=10, color='k', fontweight='normal', rotation=270, ha='center', va='center')
    plt.text(0, 1.15, r'Polar ( $J_{\phi} = 0$)', fontsize=10, color='k', fontweight='normal', rotation=0, ha='center', va='center')
    plt.text(0, -1.15, r'Radial ( $J_{\phi} = 0$)', fontsize=10, color='k', fontweight='normal', rotation=0, ha='center', va='center')

    jr   = data_init.T[4]
    jphi = data_init.T[2]
    jphi = pd.to_numeric(jphi)
    jz   = data_init.T[3]
    jtot = np.sqrt(np.array(jr, dtype=np.float32)**2 + np.array(jphi, dtype=np.float32)**2 + np.array(jz, dtype=np.float32)**2)

    plt.scatter(np.array(-jphi, dtype=np.float32)/jtot, (np.array(jz, dtype=np.float32)-np.array(jr, dtype=np.float32))/jtot,c='silver', s=5, alpha=0.3, marker='o')

    for ii in np.arange(max(labels)+1):
        jr   = np.array(data_init.T[4][(labels==ii)], dtype=np.float32)
        jphi = np.array(data_init.T[2][(labels==ii)], dtype=np.float32)
        jz   = np.array(data_init.T[3][(labels==ii)], dtype=np.float32)
        jtot = np.sqrt(jr**2 + jphi**2 + jz**2)
        jphi = pd.to_numeric(jphi)
        plt.scatter(-jphi/jtot, (jz-jr)/jtot, c=colors2[ii], s=30, alpha=1, marker=marker[ii], edgecolor='k', linewidths=0.6) 

    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.axis('off')

    #disk_jr   = disk["j_r50"  ]
    #disk_jphi = disk["j_phi50"]
    #disk_jz   = disk["j_z50"  ]
    #disk_jtot = np.sqrt(disk_jr**2 + disk_jphi**2 + disk_jz**2)

    #plt.scatter(-disk_jphi/disk_jtot, (disk_jz - disk_jr)/disk_jtot, c='steelblue' , zorder=-5, s=5, alpha=0.1, marker='o')



########### Plot8 sqrt( v_r^2 + v_z^2 ) x v_phi ###########
    ax1 = fig.add_subplot(4,4,13)

    ax1.set_axisbelow(True)
    plt.grid(b=True, lw=0.5, zorder=-2)

    plt.scatter(-np.array(data_init.T[24][(labels==-1)], dtype=np.float32)/1000, np.sqrt( np.array(data_init.T[22][(labels==-1)], dtype=np.float32)**2 + np.array(data_init.T[23][(labels==-1)], dtype=np.float32)**2)/1000, c='silver', s=5, alpha=0.3, marker='o', zorder=-2)

    ## plots the data that has been assigned to a group (color coded) ##
    for ii in np.arange(max(labels)+1):

        plt.scatter( -np.array(data_init.T[24][(labels==ii)], dtype=np.float32)/1000, np.sqrt( np.array(data_init.T[22][(labels==ii)], dtype=np.float32)**2 + np.array(data_init.T[23][(labels==ii)], dtype=np.float32)**2 )/1000, c=colors2[ii], s=30, alpha=1, marker=marker[ii], edgecolor='k', linewidths=0.6)

    plt.xlim(0, 3.0)  
    plt.ylim(0 , 4.0)
    plt.xlabel(r'$J_{\phi} = L_z$ ($\times 10^3$ km s$^{-1}$ kpc)', fontsize=15)
    plt.ylabel(r'$\sqrt{ L_x^2 + L_y^2 }$ (km s$^{-1}$)', fontsize=15)


    # Box A
    plt.vlines(1, 1.750, 2.600, colors='k', alpha=0.8, linestyles='dashed', linewidths=2, zorder=-1)
    plt.vlines(1.500, 1.750, 2.600, colors='k', alpha=0.8, linestyles='dashed', linewidths=2, zorder=-1)
    plt.hlines(1.750, 1.000, 1.500, colors='k', alpha=0.8, linestyles='dashed', linewidths=2, zorder=-1)
    plt.hlines(2.600, 1.000, 1.500, colors='k', alpha=0.8, linestyles='dashed', linewidths=2, zorder=-1)
    plt.text(1.350, 2.650 , r'A', fontsize=15, color='k', alpha=0.8, fontweight='bold')

    # Box B
    plt.vlines( 0.750, 1.600, 3.200, colors='k', alpha=0.8, linestyles='dashed', linewidths=2, zorder=-1)
    plt.vlines(1.700, 1.600, 3.200, colors='k', alpha=0.8, linestyles='dashed', linewidths=2, zorder=-1)
    plt.hlines(1.600,  0.750, 1.700, colors='k', alpha=0.8, linestyles='dashed', linewidths=2, zorder=-1)
    plt.hlines(3.200,  0.750, 1.700, colors='k', alpha=0.8, linestyles='dashed', linewidths=2, zorder=-1)
    plt.text(1.550, 3.250 , r'B', fontsize=15, color='k', alpha=0.8, fontweight='bold')


########### Plot3 ###########
    ax1 = fig.add_subplot(4,4,14) 

    ax1.set_axisbelow(True)
    plt.grid(b=True, lw=0.5, zorder=-2)

    plt.scatter(np.array(data_init.T[1][(labels==-1)], dtype=np.float32), np.array(180-data_init.T[11][(labels==-1)], dtype=np.float32), c='silver', s=5, alpha=0.3, marker='o')

    for ii in np.arange(max(labels)+1):
        plt.scatter(np.array(data_init.T[1][(labels==ii)], dtype=np.float32), np.array(180-data_init.T[11][(labels==ii)], dtype=np.float32), c=colors2[ii], s=30, alpha=1, marker=marker[ii], edgecolor='k', linewidths=0.6)  

    plt.xlim(0, 1)    
    plt.ylim(0, 180)    
    plt.xlabel(r'Eccentricity', fontsize=15)
    plt.ylabel(r'Inclination (deg)', fontsize=15)


    #plt.scatter(disk["ecc50"], 180-disk["incl50"], c='steelblue' , zorder=-5, s=5, alpha=0.1, marker='o')

########### Plot3 ###########
    ax1 = fig.add_subplot(4,4,15) 

    ax1.set_axisbelow(True)
    plt.grid(b=True, lw=0.5, zorder=-2)

    ax1.set_axisbelow(True)
    plt.grid(b=True, lw=0.5, zorder=-2)

    plt.scatter( -np.array(data_init.T[2][(labels==-1)], dtype=np.float32)/1000, np.array(data_init.T[1][(labels==-1)], dtype=np.float32), c='silver', s=5, alpha=0.3, marker='o')

    ## plots the data that has been assigned to a group (color coded) ##
    for ii in np.arange(max(labels)+1):
        plt.scatter(-np.array(data_init.T[2][(labels==ii)], dtype=np.float32)/1000, np.array(data_init.T[1][(labels==ii)], dtype=np.float32), c=colors2[ii], s=30, alpha=1, marker=marker[ii], edgecolor='k', lw=0.6)
 
    plt.ylim(0, 1)    
    plt.ylabel(r'Eccentricity', fontsize=15)
    plt.xlim(-4, 3.0)    
    plt.xlabel(r'$J_{\phi}$ ($\times 10^3$ km s$^{-1}$ kpc)', fontsize=15)

########### Plot9 Act space map ###########
    ax1 = fig.add_subplot(4,4,16)

    ax1.set_axisbelow(True)
    plt.grid(b=True, lw=0.5, zorder=-2)

    x=[-1, 0, 1, 0]
    y=[ 0, 1, 0,-1]
    
    connectpoints(x,y,0,1)
    connectpoints(x,y,1,2)
    connectpoints(x,y,2,3)
    connectpoints(x,y,3,0)

    plt.text(-1.1, 0, r'Retrograde', fontsize=10, color='k', fontweight='normal', rotation=90, ha='center', va='center')
    plt.text(1.1, 0, r'Prograde', fontsize=10, color='k', fontweight='normal', rotation=270, ha='center', va='center')
    plt.text(0, 1.1, r'Polar ( $J_{\phi} = 0$)', fontsize=10, color='k', fontweight='normal', rotation=0, ha='center', va='center')
    plt.text(0, -1.1, r'Radial ( $J_{\phi} = 0$)', fontsize=10, color='k', fontweight='normal', rotation=0, ha='center', va='center')
    plt.text(-0.58, 0.58, r'Circular ( $J_R = 0$)', fontsize=10, color='k', fontweight='normal', rotation=45, ha='center', va='center')
    plt.text(0.58, 0.58, r'Circular ( $J_R = 0$)', fontsize=10, color='k', fontweight='normal', rotation=315, ha='center', va='center')
    plt.text(0.58, -0.58, r'In plane ( $J_z = 0$)', fontsize=10, color='k', fontweight='normal', rotation=45, ha='center', va='center')
    plt.text(-0.58, -0.58, r'In plane ( $J_z = 0$)', fontsize=10, color='k', fontweight='normal', rotation=315, ha='center', va='center')

    jr   = data_init.T[4]
    jphi = data_init.T[2]
    jphi = pd.to_numeric(jphi)
    jz   = data_init.T[3]
    jtot = np.array(jr, dtype=np.float32) + abs(np.array(jphi, dtype=np.float32)) + np.array(jz, dtype=np.float32)

    plt.scatter(np.array(-jphi, dtype=np.float32)/jtot, (np.array(jz, dtype=np.float32)-np.array(jr, dtype=np.float32))/jtot,c='silver', s=5, alpha=0.3, marker='o')

    for ii in np.arange(max(labels)+1):
        jr   = np.array(data_init.T[4][(labels==ii)], dtype=np.float32)
        jphi = np.array(data_init.T[2][(labels==ii)], dtype=np.float32)
        jz   = np.array(data_init.T[3][(labels==ii)], dtype=np.float32)
        jtot = jr + abs(jphi) + jz
        jphi = pd.to_numeric(jphi)
        plt.scatter(-jphi/jtot, (jz-jr)/jtot, c=colors2[ii], s=30, alpha=1, marker=marker[ii], edgecolor='k', linewidths=0.6) 

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axis('off')


    #plt.scatter(-disk["JphiJtot"], disk["JzJrJtot"], c='steelblue' , zorder=-5, s=5, alpha=0.1, marker='o')

# save several plots showing your resulting NOMINAL clusters
    plt.subplots_adjust(wspace=0.35, hspace=0.30)
    #plt.savefig(str(method)+"_plots_"+str(k)+"_"+str(j)+"_.png", bbox_inches='tight')
     
#                      0       1   2    3    4   5   6    7    8   9    10  11    12   13  14  15     16 17 18  19    20    21   22  23  24    25
#X_init = np.vstack([energy, ecc, jphi, jz, jr, vphi, vz, vr, vx, vy, logg,incl, dist, Vr, feh, Teff, x, y, z, zmax, rapo, peri, Lx, Ly, Lz, names]).T


    test = np.c_[data_init.T[25],data_init.T[26],data_init.T[27],data_init.T[28], data_init.T[15], data_init.T[10], data_init.T[14], data_init.T[13], data_init.T[12], data_init.T[16], data_init.T[17], data_init.T[18], data_init.T[8], data_init.T[9], data_init.T[6], data_init.T[7], data_init.T[5], data_init.T[6], data_init.T[22], data_init.T[23], data_init.T[24], data_init.T[4], data_init.T[2], data_init.T[3], data_init.T[0], data_init.T[1], data_init.T[20], data_init.T[21], data_init.T[19], data_init.T[11], labels+1 ]
    
    # remove comment to save HDBSCAN output
    #np.savetxt('HDBSCAN_output.csv',test,delimiter=',', fmt='%f',header="Name_Gaia_EDR3,RA_Gaia_EDR3,DEC_Gaia_EDR3,Vmag_Gaia_EDR3,Teff,logg,FeH,RV,Distance,Xgal,Ygal,Zgal,vx,vy,vz,vr,vphi,vz,Lx,Ly,Lz,Jr,Jphi,Jz,Energy,ecc,rapo,peri,Zmax,inclination,ass_cluster",comments='')


# Save result for the monte carlo resamples
# note that this is an exquisitely dumb way of saving these results, but I ended up never making this in a better way
#print_test = np.array(print_test)
#np.savetxt('ProbCalc_.csv', print_test, delimiter=',', comments='')


###############################################################################################
# Test HDBSCAN
valores=np.arange(5,6,1) #min_samples
valores2=np.arange(5,6,1) #min_cluster_size
for (k) in valores:
    for j in valores2:
        k = float(k)
        k = int(k)
        j = float(j)
        j = int(j)
        plot_clusters(X, Y, test_X ,X_init, Y_init, hdbscan.HDBSCAN, int(j), int(k), (), {'min_cluster_size':j, 'min_samples':k})
        plt.close('all')





