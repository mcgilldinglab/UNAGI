#!/usr/bin/env python
import pdb,sys,os
from File import *
from scipy.stats import ttest_ind
import math
import json

# read in the singleCell data
ex=TabFile(sys.argv[1]).read("\t")
FR=ex[0]
ex=ex[1:]

dL={}
dT={}
for i in range(len(ex)):
    ti=ex[i][1]
    li=ex[i][2]
    if ti not in dT:
        dT[ti]=[i]
    else:
        dT[ti].append(i)

    if li not in dL:
        dL[li]=[i]
    else:
        dL[li].append(i)

GL=FR[3:]

#
rcut=200 # student T test p-value rank cutoff 
fcut=3 # fold change cutoff

out=[]
for t in dT:
    tCells=dT[t]
    for l in dL:
        lCells=dL[l]
        ov=[item for item in tCells if item in lCells]
        nov=[item for item in tCells if item not in lCells]
        lout=[]
        for g in range(len(GL)):
            xi=[float(ex[item][3+g]) for item in ov]
            yi=[float(ex[item][3+g]) for item in nov]
            fd=sum(xi)/len(xi)-sum(yi)/len(yi)
            pv=ttest_ind(xi,yi)[-1]
            lout.append([pv,fd,GL[g]])
        lout=[item for item in lout if item[1]>fcut]
        lout.sort()
        lout=lout[0:rcut]
        lout_genes=[item[-1] for item in lout]
        tlg=[t,l]+[",".join(lout_genes)]
        out.append(tlg)
        
# 
sortedCell=[] # sortedCell, we can use similar code above to create the signature gene list for sorted cells
cb=[out,sortedCell] # combined json, first element: single-cell list, second element: sorted-cell list (set it as empty list if no sorted cell data is given)

cbj=json.dumps(cb)
out="data_cells="+cbj

with open("cells.json","w") as f:
    f.write(out)

