# get top markers for each track from IDREM results, the requirements for top markers are:
# 1. the tendency of the marker is monotonic
# 2. the log2FC of the marker is larger than a cutoff
import json
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import norm
from ..dynamic_regulatory_networks.processTFs import readIdremJson


def _empty_marker_bucket(stages: int) -> dict[str, dict[str, object]]:
    bucket = {
        'gene': {},
        'log2fc': {},
        'rank': {},
    }
    for stage_idx in range(stages):
        bucket[f'stage{stage_idx}'] = {}
    bucket['qval'] = {}
    return bucket


def _vectorized_one_dist_bucket(
    *,
    temp_change: np.ndarray,
    temp_names: np.ndarray,
    temp_stages: list[np.ndarray],
    temp_background: np.ndarray,
    cutoff: float,
    increasing: bool,
) -> dict[str, dict[str, object]]:
    stages = len(temp_stages)
    bucket = _empty_marker_bucket(stages)
    if temp_change.size == 0:
        return bucket

    ordered_positions = temp_change.argsort()[::-1] if increasing else temp_change.argsort()
    flattened_background = temp_background.reshape(-1, 1)
    mean = flattened_background.mean()
    std = flattened_background.std()
    ordered_change = temp_change[ordered_positions]
    cdf = np.asarray(scoreAgainstBackground(flattened_background, ordered_change, mean=mean, std=std)).reshape(-1)
    if increasing:
        keep_mask = cdf >= (1 - cutoff)
        candidate_pvals = 1 - cdf[keep_mask]
    else:
        keep_mask = cdf <= cutoff
        candidate_pvals = cdf[keep_mask]
    candidate_positions = ordered_positions[keep_mask]
    count = int(candidate_positions.size)
    if count == 0:
        return bucket

    sort_order = np.argsort(candidate_pvals)
    pvals = candidate_pvals[sort_order]
    selected_positions = candidate_positions[sort_order]
    # Preserve the legacy ordering quirk where the already-sorted p-values are
    # sorted again before genes are materialized.
    resort = np.argsort(pvals)
    pvals = pvals[resort]
    selected_positions = selected_positions[resort]

    qvals = pvals * count / (np.arange(count) + 1)
    out_idx = 0
    for idx, q_val in zip(selected_positions, qvals):
        if q_val > cutoff:
            continue
        key = str(out_idx)
        bucket['gene'][key] = temp_names[idx]
        bucket['log2fc'][key] = temp_change[idx]
        bucket['rank'][key] = out_idx + 1
        bucket['qval'][key] = q_val
        for stage_idx in range(stages):
            bucket[f'stage{stage_idx}'][key] = temp_stages[stage_idx][idx]
        out_idx += 1
    return bucket


def _getTopMarkers_one_dist_optimized(idrem, background, filename, cutoff):
    topMarkers = {}
    tt = readIdremJson(idrem, filename)

    background = np.asarray(background)
    stages = len(filename.split('.')[0].split('-'))
    temp = np.array(tt[8])
    idrem_genes = np.array(temp[1:, 0].tolist())
    tendency = temp[1:, 1].astype(float)
    for i in range(2, stages):
        tendency *= temp[1:, i].astype(float)
    tendency[tendency < 0] = 0
    index = [i for i, x in enumerate(tendency) if x <= 0]
    change = temp[1:, stages].astype(float) - temp[1:, 1].astype(float)
    change[index] = 0
    stage_values = [temp[1:, i + 1].astype(float) for i in range(stages)]

    increasing_stop = np.where(change > 0)[0]
    decreasing_stop = np.where(change < 0)[0]

    topMarkers['increasing'] = _vectorized_one_dist_bucket(
        temp_change=change[increasing_stop],
        temp_names=idrem_genes[increasing_stop],
        temp_stages=[values[increasing_stop] for values in stage_values],
        temp_background=background[:, increasing_stop],
        cutoff=float(cutoff),
        increasing=True,
    )
    topMarkers['decreasing'] = _vectorized_one_dist_bucket(
        temp_change=change[decreasing_stop],
        temp_names=idrem_genes[decreasing_stop],
        temp_stages=[values[decreasing_stop] for values in stage_values],
        temp_background=background[:, decreasing_stop],
        cutoff=float(cutoff),
        increasing=False,
    )
    return topMarkers
def scoreAgainstBackground(background,input,all=False,mean=None,std=None):
    '''
    Calculate the p-value of the input gene expression change based on the background gene expression change.
    
    parameters
    --------------------
    background: the background gene expression change
    input: the input gene expression change
    all: if all is True, the input gene expression change will be compared with all the background gene expression changes. Otherwise, the input gene expression change will be compared with each background gene expression change.
    mean: the mean of the background gene expression change
    std: the standard deviation of the background gene expression change

    return
    --------------------
    cdf: np.float
        p-value of the input gene expression change
    '''

    if all:
        background = background.reshape(-1,1)

    # print(background.mean(axis=0), background.std(axis=0))
    if mean is None:
        cdf = norm.cdf(input, loc=background.mean(axis=0), scale=background.std(axis=0))
    else:
        cdf = norm.cdf(input, loc=mean, scale=std)
    
    return cdf

def getTopMarkers(idrem, background,filename, cutoff=None,topN=None,one_dist=False, strategy='legacy'):
    '''
    **not updated**
    '''
    topMarkers = {}
    tt = readIdremJson(idrem,filename)

    background = np.array(background)
    stages = len(filename.split('.')[0].split('-'))
    
    temp = np.array(tt[8])
    idrem_genes = np.array(temp[1:,0].tolist())
    tendency = temp[1:,1].astype(float)
    for i in range(2,stages):
        tendency*=temp[1:,i].astype(float)
    tendency[tendency <0] = 0
    index = [i for i, x in enumerate(tendency) if x <= 0]
    change = temp[1:,stages].astype(float) - temp[1:,1].astype(float)
    stage_values = []
    for i in range(0,stages):
        stage_values.append(temp[1:,i+1].astype(float))

    stages_list = []
    for i in range(0,stages):
        stages_list.append(temp[1:,i+1].astype(float))
    
    change[index] = 0
    if cutoff is not None:
        topMarkers['increasing'] = {}
        topMarkers['decreasing'] = {}
        increasing_stop = np.where(change > 0)[0]
        decreasing_stop = np.where(change < 0)[0]
        temp_change = change[increasing_stop]
        temp_names = idrem_genes[increasing_stop]
        temp_stages = []
        for i in range(0,stages):
            temp_stages.append(stages_list[i][increasing_stop])
        temp_background = background[:,increasing_stop]

        
        increasing_stop = temp_change.argsort()[::-1]
        # increasing_pval = scoreAgainstBackground(background[:,increasing_stop],temp_change)
        
        topMarkers['increasing']['gene'] = {}
        topMarkers['increasing']['log2fc'] = {}
        topMarkers['increasing']['rank'] = {}
        for i in range(0,stages):
            topMarkers['increasing']['stage'+str(i)] = {} 
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
            for i in range(0,stages):
                topMarkers['increasing']['stage'+str(i)][str(count_1)] = temp_stages[i][each]
            topMarkers['increasing']['log2fc'][str(count_1)] = temp_change[each]
            topMarkers['increasing']['rank'][str(count_1)] = count_1+1
            topMarkers['increasing']['qval'][str(count_1)] = q_val
            count_1+=1
        temp_change = change[decreasing_stop]
        temp_names = idrem_genes[decreasing_stop]
        temp_stages = []
        for i in range(0,stages):
            temp_stages.append(stages_list[i][decreasing_stop]) 
        temp_background = background[:,decreasing_stop]
        decreasing_stop = temp_change.argsort()
        # decreasing_pval = scoreAgainstBackground(background[:,decreasing_stop],temp_change)
        
        topMarkers['decreasing']['gene'] = {}
        topMarkers['decreasing']['log2fc'] = {}
        topMarkers['decreasing']['rank'] = {}
        for i in range(0,stages):
            topMarkers['decreasing']['stage'+str(i)] = {}
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
            for i in range(0,stages):
                topMarkers['decreasing']['stage'+str(i)][str(count_1)] = temp_stages[i][each]
            topMarkers['decreasing']['qval'][str(count_1)] = q_val
            count_1+=1
    else: # if cutoff is not given, return ranked list
        topMarkers['increasing'] = {}
        topMarkers['decreasing'] = {}
        increasing_stop = np.where(change > 0)[0]
        decreasing_stop = np.where(change < 0)[0]
        temp_change = change[increasing_stop]
        temp_names = idrem_genes[increasing_stop]
        temp_stages = []
        for i in range(0,stages):
            temp_stages.append(stages_list[i][increasing_stop])
        increasing_stop = temp_change.argsort()[::-1]
        topMarkers['increasing']['gene'] = {}
        topMarkers['increasing']['log2fc'] = {}
        topMarkers['increasing']['rank'] = {}
        for i in range(0,stages):
            topMarkers['increasing']['stage'+str(i)] = {}
        if topN is not None:
            for i, each in enumerate(increasing_stop[:topN]):
                topMarkers['increasing']['gene'][str(i)] = temp_names[each]
                for i in range(0,stages):
                    topMarkers['increasing']['stage'+str(i)][str(i)] = temp_stages[i][each]
                topMarkers['increasing']['log2fc'][str(i)] = temp_change[each]
                topMarkers['increasing']['rank'][str(i)] = i+1
        else:
            for i, each in enumerate(increasing_stop):
                topMarkers['increasing']['gene'][str(i)] = temp_names[each]
                for i in range(0,stages):
                    topMarkers['increasing']['stage'+str(i)][str(i)] = temp_stages[i][each]
                topMarkers['increasing']['log2fc'][str(i)] = temp_change[each]
                topMarkers['increasing']['rank'][str(i)] = i+1
        temp_change = change[decreasing_stop]
        temp_names = idrem_genes[decreasing_stop]
        temp_stages = []
        for i in range(0,stages):
            temp_stages.append(stages_list[i][decreasing_stop])
        decreasing_stop = temp_change.argsort()
        topMarkers['decreasing']['gene'] = {}
        topMarkers['decreasing']['log2fc'] = {}
        topMarkers['decreasing']['rank'] = {}
        for i in range(0,stages):
            topMarkers['decreasing']['stage'+str(i)] = {}
        if topN is not None:
            for i, each in enumerate(decreasing_stop[:topN]):
                topMarkers['decreasing']['gene'][str(i)] = temp_names[each]
                topMarkers['decreasing']['log2fc'][str(i)] = temp_change[each]
                for i in range(0,stages):
                    topMarkers['decreasing']['stage'+str(i)][str(i)] = temp_stages[i][each]
                topMarkers['decreasing']['rank'][str(i)] = i+1

        else:
            for i, each in enumerate(decreasing_stop):
                topMarkers['decreasing']['gene'][str(i)] = temp_names[each]
                topMarkers['decreasing']['log2fc'][str(i)] = temp_change[each]
                topMarkers['decreasing']['rank'][str(i)] = i+1 
                for i in range(0,stages):
                    topMarkers['decreasing']['stage'+str(i)][str(i)] = temp_stages[i][each]


    return topMarkers
def getTopMarkersFromIDREM(path, background,cutoff=None,topN=None,one_dist=False, strategy='legacy'):
    '''
    Get the top markers for each track from IDREM results.
    parameters
    --------------------
    path: str
        the directory to the IDREM results.
    background: np.array
        the background gene expression change
    cutoff: float
        the cutoff for p-value. Default is None.
    topN: int
        the number of top markers to return. Default is None.
    one_dist: bool
        whether to consider all the background gene expression changes as one distribution. Default is False.

    return
    ----------------
    out:    dict
        a dictionary of top markers for each track.
    '''
    out = {}
    filenames = os.listdir(path)
    for each in filenames:
        name = each
        if each[0] != '.':
            each = each.split('.')[0]#.split('-')[-1].split('n')
            if one_dist:
                out[each] = getTopMarkers(path,background,name,cutoff,topN,one_dist=one_dist, strategy=strategy)
            else:
                out[each] = getTopMarkers(path,background[each],name,cutoff,topN, strategy=strategy)
    return out

def runGetProgressionMarkercsv(directory,background, save_dir, topN=None,cutoff=None, strategy='legacy'):
    '''
    Get the top markers for each track from IDREM results and save as a csv file.
    
    parameters
    --------------------
    directory: str
        the directory to the IDREM results.
    background: str
        the directory to the background gene expression change.
    cutoff: float
        the cutoff for p-value. Default is None.
    topN: int
        the number of top markers to return. Default is None.
    save_dir: str
        the directory to save the csv file.

    return
    ---------------
    None
    '''
    from pathlib import Path
    save_dir = Path(save_dir)
    background = dict(np.load(background,allow_pickle=True).tolist())
    out = getTopMarkersFromIDREM(directory,background,topN=topN,cutoff=cutoff, strategy=strategy)
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
    # return results
    results.to_csv(save_dir/'mesProgressionMarker_pval_twofilters.csv')


def runGetProgressionMarker(directory,background, cutoff=0.05, topN=None, strategy='legacy'):
    '''
    Get the top markers for each track from IDREM results.
    
    parameters
    --------------------
    directory: str
        the directory to the IDREM results.
    background: str
        the directory to the background gene expression change.
    cutoff: float
        the cutoff for p-value. Default is 0.05.
    topN: int
        the number of top markers to return. Default is None.

    return
    ---------------
    out:    dict
        a dictionary of top markers for each track.
    '''
    out = getTopMarkersFromIDREM(directory,background,cutoff = cutoff,topN=topN, strategy=strategy)
    return out


def runGetProgressionMarker_one_dist(directory,background,size, cutoff=0.05, topN=None, strategy='optimized'):
    '''
    Get the top markers for each track from IDREM results and consider the whole background as one distribution.
    
    parameters
    --------------------
    directory: str
        the directory to the IDREM results.
    background: str
        the directory to the background gene expression change.
    cutoff: float
        the cutoff for p-value. Default is 0.05.
    topN: int
        the number of top markers to return. Default is None.

    return
    ---------------
    out:    
        a dictionary of top markers for each track.
    '''

    from ..utils.progress_logger import ProgressLogger
    logger = ProgressLogger(prefix='[UNAGI-PROGRESSION]')
    started_at = time.time()
    logger.log(f"start marker_identification tracks={len(background)} genes={size} cutoff={cutoff} strategy={strategy}")
    one_dist  = []
    for each in background.keys(): 
        one_dist.append(np.array(background[each]))
    background = np.array(one_dist).reshape(-1, size)
    out = getTopMarkersFromIDREM(directory,background,cutoff = cutoff,one_dist=True, strategy=strategy)
    logger.log(
        f"finished marker_identification elapsed={ProgressLogger._format_duration(time.time() - started_at)}"
    )
    return out
