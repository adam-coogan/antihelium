import sys, math, csv
from matplotlib import pyplot as plt
import numpy as np



def plotAntideuterons(filename):
    
    
    CMS,A,Z,E = [], [], [], []
    
    
    f = open(filename, 'r')
    csvread = csv.reader(open(filename, 'rb'),delimiter = ' ' )
    
    
    for line in csvread:
        if 'RUNDETAILS' in line:
            continue
        CMS.append(float(line[0]))
        A.append(float(line[1]))
        Z.append(float(line[2]))
        E.append(float(line[3]))
    
    # Get a list of each different mass in the file   
    masses = np.sort(np.array(list(set(CMS))))
    
    bins = np.logspace(0, 2, 21)
    dbar,he3bar,tbar,he4bar = [],[],[],[]
    for i in range(len(masses)):
        print '\nDM Mass ', masses[i]/2., ' GeV'
        dbar.append([])
        he3bar.append([])
        tbar.append([])
        he4bar.append([])
        
        indices = [j for j, mass in enumerate(CMS) if mass == masses[i]]
        
        for idx in indices:
            if A[idx] == 2 and Z[idx] == 1: dbar[i].append(E[idx])
            if A[idx] == 3 and Z[idx] == 1: tbar[i].append(E[idx])
            if A[idx] == 3 and Z[idx] == 2: he3bar[i].append(E[idx])
            if A[idx] == 4 and Z[idx] == 2: he4bar[i].append(E[idx])
        
        print 'Found ', len(dbar[i]), ' antideuterons.'
        print 'Found ', len(tbar[i]), ' antitritons.'
        print 'Found ', len(he3bar[i]), ' antihelium 3.'
        print 'Found ', len(he4bar[i]), ' antihelium 4.'
    
        values, bins = np.histogram(dbar[i], bins = bins)
        for x in range(len(values)):
            values[i]/= (bins[i+1]-bins[i]) 
        plt.step(bins[:-1], values,label = str(masses[i]/2.)+' GeV')
        
    plt.ylabel('dN/dE (count/GeV)')
    plt.xlabel('E (GeV)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()
     
plotAntideuterons(sys.argv[1])