# get top markers for each track from IDREM results, the requirements for top markers are:
# 1. the tendency of the marker is monotonic
# 2. the log2FC of the marker is larger than a cutoff
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
def scoreAgainstBackground(background,input,all=False,mean=None,std=None):
    #calculate pval for each gene
    #input is a list chages for all genes
    if all:
        background = background.reshape(-1,1)

    # print(background.mean(axis=0), background.std(axis=0))
    if mean is None:
        cdf = norm.cdf(input, loc=background.mean(axis=0), scale=background.std(axis=0))
    else:
        cdf = norm.cdf(input, loc=mean, scale=std)
    
    return cdf

def readIdremJson(path, filename):
    print('getting Target genes from ', filename)

    path = os.path.join(path,filename,'DREM.json')
    f=open(path,"r")
    lf=f.readlines()
    f.close()
    lf="".join(lf)
    lf=lf[5:-2]+']'
    tt=json.loads(lf,strict=False)
    return tt
def getTopMarkers(idrem, background,filename, cutoff=None,topN=None,one_dist=False):
    topMarkers = {}
    tt = readIdremJson(idrem,filename)

    background = np.array(background)
    
    
    temp = np.array(tt[8])
    idrem_genes = np.array(temp[1:,0].tolist())
    tendency = temp[1:,4].astype(float)* temp[1:,3].astype(float) * temp[1:,2].astype(float) * temp[1:,1].astype(float)
    tendency[tendency <0] = 0
    index = [i for i, x in enumerate(tendency) if x <= 0]
    change = temp[1:,4].astype(float) - temp[1:,1].astype(float)
    stage0 = temp[1:,1].astype(float)
    stage1 = temp[1:,2].astype(float)
    stage2 = temp[1:,3].astype(float)
    stage3 = temp[1:,4].astype(float)
    
    change[index] = 0
    if cutoff is not None:
        topMarkers['increasing'] = {}
        topMarkers['decreasing'] = {}
        increasing_stop = np.where(change > 0)[0]
        decreasing_stop = np.where(change < 0)[0]
        temp_change = change[increasing_stop]
        temp_names = idrem_genes[increasing_stop]
        temp_stage0 = stage0[increasing_stop]
        temp_stage1 = stage1[increasing_stop]
        temp_stage2 = stage2[increasing_stop]
        temp_stage3 = stage3[increasing_stop]
        temp_background = background[:,increasing_stop]

        
        increasing_stop = temp_change.argsort()[::-1]
        # increasing_pval = scoreAgainstBackground(background[:,increasing_stop],temp_change)
        
        topMarkers['increasing']['gene'] = {}
        topMarkers['increasing']['log2fc'] = {}
        topMarkers['increasing']['rank'] = {}
        topMarkers['increasing']['stage0'] = {}
        topMarkers['increasing']['stage1'] = {}
        topMarkers['increasing']['stage2'] = {}
        topMarkers['increasing']['stage3'] = {}
        topMarkers['increasing']['qval'] = {}
        count = 0
        pvals = []
        in_pvals = []
        if one_dist:
            temp_background = temp_background.reshape(-1,1)
            mean = temp_background.mean()
            std = temp_background.std()
        for i, each in enumerate(increasing_stop):
            if one_dist:
                increasing_pval = scoreAgainstBackground(temp_background,temp_change[each],mean = mean, std = std)
            else:
                increasing_pval = scoreAgainstBackground(temp_background[:, each],temp_change[each])
                if increasing_pval < (1-cutoff):
                    continue
                increasing_pval = scoreAgainstBackground(temp_background,temp_change[each],all=True)
            if increasing_pval < (1-cutoff):
                continue
            in_pvals.append(each)
            pvals.append(1-increasing_pval)
            # topMarkers['increasing']['gene'][str(count)] = temp_names[each]
            # topMarkers['increasing']['stage0'][str(count)] = temp_stage0[each]
            # topMarkers['increasing']['stage1'][str(count)] = temp_stage1[each]
            # topMarkers['increasing']['stage2'][str(count)] = temp_stage2[each]
            # topMarkers['increasing']['stage3'][str(count)] = temp_stage3[each]
            # topMarkers['increasing']['log2fc'][str(count)] = temp_change[each]
            # topMarkers['increasing']['rank'][str(count)] = count+1
            # topMarkers['increasing']['pval'][str(count)] = 1-increasing_pval#[each]
            count+=1
        count_1 = 0
        #sort pvals by sorted
        pvals = np.array(pvals).reshape(-1)

        in_pvals = np.array(in_pvals).reshape(-1)
        pvals = pvals[pvals.argsort()]

        in_pvals = in_pvals[pvals.argsort()]
        for i in range(count):
            
            q_val = pvals[i] * count / (i+1)
            if q_val > cutoff:
                continue
            each = in_pvals[i]
            topMarkers['increasing']['gene'][str(count_1)] = temp_names[each]
            topMarkers['increasing']['stage0'][str(count_1)] = temp_stage0[each]
            topMarkers['increasing']['stage1'][str(count_1)] = temp_stage1[each]
            topMarkers['increasing']['stage2'][str(count_1)] = temp_stage2[each]
            topMarkers['increasing']['stage3'][str(count_1)] = temp_stage3[each]
            topMarkers['increasing']['log2fc'][str(count_1)] = temp_change[each]
            topMarkers['increasing']['rank'][str(count_1)] = count_1+1
            topMarkers['increasing']['qval'][str(count_1)] = q_val
            count_1+=1
        temp_change = change[decreasing_stop]
        temp_names = idrem_genes[decreasing_stop]
        temp_stage0 = stage0[decreasing_stop]
        temp_stage1 = stage1[decreasing_stop]
        temp_stage2 = stage2[decreasing_stop]
        temp_stage3 = stage3[decreasing_stop]
        temp_background = background[:,decreasing_stop]
        decreasing_stop = temp_change.argsort()
        # decreasing_pval = scoreAgainstBackground(background[:,decreasing_stop],temp_change)
        
        topMarkers['decreasing']['gene'] = {}
        topMarkers['decreasing']['log2fc'] = {}
        topMarkers['decreasing']['rank'] = {}
        topMarkers['decreasing']['stage0'] = {}
        topMarkers['decreasing']['stage1'] = {}
        topMarkers['decreasing']['stage2'] = {}
        topMarkers['decreasing']['stage3'] = {}
        topMarkers['decreasing']['qval'] = {}
        count = 0
        pvals = []
        in_pvals = []
        if one_dist:
            temp_background = temp_background.reshape(-1,1)
            mean = temp_background.mean()
            std = temp_background.std()
        for i, each in enumerate(decreasing_stop):
            if one_dist:
                decreasing_pval = scoreAgainstBackground(temp_background,temp_change[each],mean = mean, std = std)
            else:
                decreasing_pval = scoreAgainstBackground(temp_background[:, each],temp_change[each])
                if decreasing_pval > cutoff:
                    continue
                decreasing_pval = scoreAgainstBackground(temp_background,temp_change[each],all=True)
            if decreasing_pval > cutoff:
                continue
            in_pvals.append(each)
            pvals.append(decreasing_pval)
            # topMarkers['decreasing']['gene'][str(count)] = temp_names[each]
            # topMarkers['decreasing']['log2fc'][str(count)] = temp_change[each]
            # topMarkers['decreasing']['rank'][str(count)] = count+1 
            # topMarkers['decreasing']['stage0'][str(count)] = temp_stage0[each]
            # topMarkers['decreasing']['stage1'][str(count)] = temp_stage1[each]
            # topMarkers['decreasing']['stage2'][str(count)] = temp_stage2[each]
            # topMarkers['decreasing']['stage3'][str(count)] = temp_stage3[each]
            # topMarkers['decreasing']['pval'][str(count)] = decreasing_pval#[each]
            count+=1
        count_1 = 0
        
        pvals = np.array(pvals).reshape(-1)
        in_pvals = np.array(in_pvals).reshape(-1)
        pvals = pvals[pvals.argsort()]
        in_pvals = in_pvals[pvals.argsort()]

        for i in range(count):
            q_val = pvals[i] * count/(i+1)  #1 - (1 - topMarkers['decreasing']['pval'][str(i)]) * (count/(i+1))
            if q_val > cutoff:
                continue
            each = in_pvals[i]
            topMarkers['decreasing']['gene'][str(count_1)] = temp_names[each]
            topMarkers['decreasing']['log2fc'][str(count_1)] = temp_change[each]
            topMarkers['decreasing']['rank'][str(count_1)] = count_1+1 
            topMarkers['decreasing']['stage0'][str(count_1)] = temp_stage0[each]
            topMarkers['decreasing']['stage1'][str(count_1)] = temp_stage1[each]
            topMarkers['decreasing']['stage2'][str(count_1)] = temp_stage2[each]
            topMarkers['decreasing']['stage3'][str(count_1)] = temp_stage3[each]
            topMarkers['decreasing']['qval'][str(count_1)] = q_val
            count_1+=1
    else: # if cutoff is not given, return ranked list
        topMarkers['increasing'] = {}
        topMarkers['decreasing'] = {}
        increasing_stop = np.where(change > 0)[0]
        decreasing_stop = np.where(change < 0)[0]
        temp_change = change[increasing_stop]
        temp_names = idrem_genes[increasing_stop]
        temp_stage0 = stage0[increasing_stop]
        temp_stage1 = stage1[increasing_stop]
        temp_stage2 = stage2[increasing_stop]
        temp_stage3 = stage3[increasing_stop]
        increasing_stop = temp_change.argsort()[::-1]
        topMarkers['increasing']['gene'] = {}
        topMarkers['increasing']['log2fc'] = {}
        topMarkers['increasing']['rank'] = {}
        topMarkers['increasing']['stage0'] = {}
        topMarkers['increasing']['stage1'] = {}
        topMarkers['increasing']['stage2'] = {}
        topMarkers['increasing']['stage3'] = {}
        if topN is not None:
            for i, each in enumerate(increasing_stop[:topN]):
                topMarkers['increasing']['gene'][str(i)] = temp_names[each]
                topMarkers['increasing']['stage0'][str(i)] = temp_stage0[each]
                topMarkers['increasing']['stage1'][str(i)] = temp_stage1[each]
                topMarkers['increasing']['stage2'][str(i)] = temp_stage2[each]
                topMarkers['increasing']['stage3'][str(i)] = temp_stage3[each]
                topMarkers['increasing']['log2fc'][str(i)] = temp_change[each]
                topMarkers['increasing']['rank'][str(i)] = i+1
        else:
            for i, each in enumerate(increasing_stop):
                topMarkers['increasing']['gene'][str(i)] = temp_names[each]
                topMarkers['increasing']['stage0'][str(i)] = temp_stage0[each]
                topMarkers['increasing']['stage1'][str(i)] = temp_stage1[each]
                topMarkers['increasing']['stage2'][str(i)] = temp_stage2[each]
                topMarkers['increasing']['stage3'][str(i)] = temp_stage3[each]
                topMarkers['increasing']['log2fc'][str(i)] = temp_change[each]
                topMarkers['increasing']['rank'][str(i)] = i+1
        temp_change = change[decreasing_stop]
        temp_names = idrem_genes[decreasing_stop]
        temp_stage0 = stage0[decreasing_stop]
        temp_stage1 = stage1[decreasing_stop]
        temp_stage2 = stage2[decreasing_stop]
        temp_stage3 = stage3[decreasing_stop]
        decreasing_stop = temp_change.argsort()
        topMarkers['decreasing']['gene'] = {}
        topMarkers['decreasing']['log2fc'] = {}
        topMarkers['decreasing']['rank'] = {}
        topMarkers['decreasing']['stage0'] = {}
        topMarkers['decreasing']['stage1'] = {}
        topMarkers['decreasing']['stage2'] = {}
        topMarkers['decreasing']['stage3'] = {}
        if topN is not None:
            for i, each in enumerate(decreasing_stop[:topN]):
                topMarkers['decreasing']['gene'][str(i)] = temp_names[each]
                topMarkers['decreasing']['log2fc'][str(i)] = temp_change[each]
                topMarkers['decreasing']['stage0'][str(i)] = temp_stage0[each]
                topMarkers['decreasing']['stage1'][str(i)] = temp_stage1[each]
                topMarkers['decreasing']['stage2'][str(i)] = temp_stage2[each]
                topMarkers['decreasing']['stage3'][str(i)] = temp_stage3[each]
                topMarkers['decreasing']['rank'][str(i)] = i+1

        else:
            for i, each in enumerate(decreasing_stop):
                topMarkers['decreasing']['gene'][str(i)] = temp_names[each]
                topMarkers['decreasing']['log2fc'][str(i)] = temp_change[each]
                topMarkers['decreasing']['rank'][str(i)] = i+1 
                topMarkers['decreasing']['stage0'][str(i)] = temp_stage0[each]
                topMarkers['decreasing']['stage1'][str(i)] = temp_stage1[each]
                topMarkers['decreasing']['stage2'][str(i)] = temp_stage2[each]
                topMarkers['decreasing']['stage3'][str(i)] = temp_stage3[each]


    return topMarkers
def getTopMarkersFromIDREM(path, background,cutoff=None,topN=None,one_dist=False):
    out = {}
    filenames = os.listdir(path)
    for each in filenames:
        name = each
        if each[0] != '.':
            each = each.split('.')[0]#.split('-')[-1].split('n')
            if one_dist:
                out[each] = getTopMarkers(path,background,name,cutoff,topN,one_dist=one_dist)
            else:
                out[each] = getTopMarkers(path,background[each],name,cutoff,topN)
            # print(out[each]['decreasing'])
            # df = pd.DataFrame.from_dict(out[each], orient='index')
    return out

def runGetProgressionMarkercsv(directory,background, topN=None,cutoff=None):
    background = dict(np.load(background,allow_pickle=True).tolist())
    out = getTopMarkersFromIDREM(directory,background,topN=topN,cutoff=cutoff)
    results = pd.json_normalize(out)
    results.columns = results.columns.str.split(".").map(tuple)
    results = results.stack([0,1,2]).reset_index(0, drop=True)
    results = results.transpose()
    results = results.reset_index()
    results['index'] = results['index'].astype('int32')
    results = results.set_index(['index']).sort_index()
    # results = results.transpose()

    results = results.sort_index()
    # print(results)
    return results
    # results.to_csv('mesProgressionMarker_pval_twofilters.csv') 


def runGetProgressionMarker(directory,background, cutoff=0.05, topN=None):
    out = getTopMarkersFromIDREM(directory,background,cutoff = cutoff)
    return out
# df = pd.DataFrame.from_dict({(i,j,k): out[i][j][k]

                    #        for i in out.keys()  
                    #        for j in out[i].keys()
                    #        for k in out[i][j].keys()},
                    #    orient='index')
    # results = pd.json_normalize(out)
    # results.columns = results.columns.str.split(".").map(tuple)
    # results = results.stack([0,1,2]).reset_index(0, drop=True)
    # results = results.transpose()
    # results = results.reset_index()
    # results['index'] = results['index'].astype('int32')
    # results = results.set_index(['index']).sort_index()
    # results = results.transpose()
    
    # results = results.sort_index()
    # results.to_csv('kankan.csv') 
    # print(results.index.tolist())

def runGetProgressionMarker_one_dist(directory,background,size, cutoff=0.05, topN=None):
    one_dist  = []
    for each in background.keys(): 
        one_dist.append(np.array(background[each]))
    background = np.array(one_dist).reshape(-1, size)
    out = getTopMarkersFromIDREM(directory,background,cutoff = cutoff,one_dist=True)
    return out