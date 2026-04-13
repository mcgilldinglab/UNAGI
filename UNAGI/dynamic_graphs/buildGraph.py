import numpy as np
import scanpy as sc
import gc
import os
from pathlib import Path
from .distDistance import *


def _candidate_edge_stats(distances):
    """
    Summarize the nearest-cluster candidate for each target-stage cluster.
    """
    distance_array = np.asarray(distances, dtype=float)
    if distance_array.ndim == 1:
        distance_array = distance_array.reshape(1, -1)

    flat = distance_array.ravel()
    loc = float(np.mean(flat))
    scale = float(np.std(flat))
    candidates = []
    for target_cluster, row in enumerate(distance_array):
        source_cluster = int(np.argmin(row))
        best_distance = float(row[source_cluster])
        if scale > 0:
            pval = float(norm.cdf(best_distance, loc=loc, scale=scale))
        else:
            pval = 0.5 if best_distance == loc else float("nan")
        candidates.append(
            {
                "source_cluster": source_cluster,
                "target_cluster": target_cluster,
                "distance": best_distance,
                "pval": pval,
            }
        )
    return candidates, loc, scale


def _format_no_edges_error(stage1, stage2, distances, cutoff):
    """
    Build a precise error message when every candidate edge is filtered out.
    """
    candidates, loc, scale = _candidate_edge_stats(distances)
    if not candidates:
        return (
            f"No dynamic graph edge candidates could be evaluated between stage {stage1} "
            f"and stage {stage2} with connect_edges_cutoff={cutoff}."
        )

    finite_candidates = [candidate for candidate in candidates if np.isfinite(candidate["pval"])]
    if finite_candidates:
        best_candidate = min(finite_candidates, key=lambda candidate: candidate["pval"])
        min_pval = min(candidate["pval"] for candidate in finite_candidates)
        max_pval = max(candidate["pval"] for candidate in finite_candidates)
        return (
            f"No dynamic graph edges passed connect_edges_cutoff={cutoff} between stage {stage1} "
            f"and stage {stage2}. All nearest-cluster candidate p-values were above the cutoff "
            f"(min_pval={min_pval:.6f}, max_pval={max_pval:.6f}). Best candidate was stage {stage1} "
            f"cluster {best_candidate['source_cluster']} -> stage {stage2} cluster "
            f"{best_candidate['target_cluster']} with distance={best_candidate['distance']:.6f} "
            f"and pval={best_candidate['pval']:.6f}. Distance distribution summary: "
            f"mean={loc:.6f}, std={scale:.6f}. Increase run.connect_edges_cutoff or adjust the "
            f"neighbor / clustering settings if you want more permissive graph connections."
        )

    return (
        f"No dynamic graph edges passed connect_edges_cutoff={cutoff} between stage {stage1} "
        f"and stage {stage2}. Candidate p-values were non-finite because the distance distribution "
        f"collapsed (mean={loc:.6f}, std={scale:.6f}). Adjust run.connect_edges_cutoff or the "
        f"neighbor / clustering settings before retrying."
    )


def nodesDistance(rep1,rep2,topgene1,topgene2):
    '''
    calculate the distance between two stages
    
    parameters
    -------------------
    rep1: list
        The representation of clusters in stage 1
    rep2: list
        The representation of clusters in stage 2
    topgene1: list
        Top 100 differential gene of clusters in stage 1
    topgene2: list
        Top 100 differential gene of clusters in stage 2
    
    return
    -------------------
    distance: list
        A list of normalized distance of clusters between two stages
    '''
    distance = [[] for _ in range(len(rep2))]
    for i in range(len(rep2)):
        for j in range(len(rep1)):
            gaussiankl_1 = calculateKL(rep2[i],rep1[j])
            gaussiankl_2 = calculateKL(rep1[j],rep2[i])
            gaussiankl = (gaussiankl_1 + gaussiankl_2)/2
            similarityDE = getSimilarity(topgene2,topgene1,i,j)
            distance[i].append([gaussiankl , similarityDE])
    for i in range(len(distance)):
        distance[i] = normalizeDistance(distance[i])
    return distance
def connectNodes(distances,cutoff = 0.05):
    '''
    Connect the clusters in two stages with smallest distance and p-value < cut-off

    parameters
    -------------------
    distances: list
        The list of distance between two stages
    cutoff: float
        The cutoff of p-value

    return
    -------------------
    edges: list
        The edges between two stages
    '''
    edges = []
    candidates, _, _ = _candidate_edge_stats(distances)
    for candidate in candidates:
        if np.isfinite(candidate["pval"]) and candidate["pval"] < cutoff:
            edges.append([candidate["source_cluster"], candidate["target_cluster"]])
    return edges

def buildEdges(stage1,stage2,cutoff = 0.05):
    '''
    calculate the distance between two stages and connect the clusters in two stages with smallest distance
    
    parameters
    -------------------

    stage1: anndata
        The data of the first selected stage
    stage2: anndata
        The data of the second selected stage
    cutoff: float
        The cutoff of p-value
    
    return
    -------------------
    edges: list
        The edges between two stages
    '''
    adata1 = sc.read_h5ad(Path('stagedata') / f'{stage1}.h5ad')
    adata2 = sc.read_h5ad(Path('stagedata') / f'{stage2}.h5ad')
    reps = np.load(Path('stagedata') / 'rep.npy', allow_pickle=True)
    
    rep1 = reps[stage1]
    rep2 = reps[stage2]
    topgene1 = adata1.uns['topGene']
    topgene2 = adata2.uns['topGene']
    distance = nodesDistance(rep1,rep2,topgene1,topgene2)
    edges = connectNodes(distance,cutoff)
    if len(edges) == 0:
        raise RuntimeError(_format_no_edges_error(stage1, stage2, distance, cutoff))
    return edges
def buildEdges(stage1,stage2,midpath,iteration,cutoff = 0.05):
    '''
    calculate the distance between two stages and connect the clusters in two stages with smallest distance with midpath in iterative training
    
    parameters
    ------------------- 

    midpath: str
        The task name
    iteration: int
        The iteration of the training
    stage1: anndata
        The data of the first selected stage
    stage2: anndata
        The data of the second selected stage
    cutoff: float
        The cutoff of p-value

    return
    -------------------
    edges: list
        The edges between two stages
    '''
    midpath = Path(midpath)
    # print(stage1,stage2)
    adata1 = sc.read_h5ad(midpath / str(iteration) / 'stagedata' / f'{stage1}.h5ad')
    adata2 = sc.read_h5ad(midpath / str(iteration) / 'stagedata' / f'{stage2}.h5ad')
    reps = np.load(midpath / str(iteration) / 'stagedata' / 'rep.npy', allow_pickle=True)

    rep1 = reps[stage1]
    rep2 = reps[stage2]
    topgene1 = adata1.uns['topGene']

    topgene2 = adata2.uns['topGene']
    distance = nodesDistance(rep1,rep2,topgene1,topgene2)
    edges = connectNodes(distance,cutoff)
    if len(edges) == 0:
        raise RuntimeError(_format_no_edges_error(stage1, stage2, distance, cutoff))
    return edges
def getandUpadateEdges(total_stage,midpath,iteration,cutoff = 0.05):
    '''
    get edges in iterative training.
    
    parameters
    -------------------
    total_stage: int
        The total number of stages
    midpath: str
        The task name
    iteration: int
        The iteration of the training
    cutoff: float
        The cutoff of p-value to connect nodes

    return
    -------------------
    edges: list
        The edges between two stages
    '''
    edges = []
    for i in range(total_stage-1):
        edges.append(buildEdges(i,i+1,midpath,iteration,cutoff=cutoff))
    edges = updateEdges(edges,midpath,iteration)
    return edges

def updateEdges(edges,midpath,iteration):
    '''
    updata edges to the anndata database, calculate edges changes.

    parameters
    -------------------
    edges: list
        The edges between two stages
    midpath: str
        The task name
    iteration: int
        The iteration of the training

    return
    -------------------
    edges: list
        The edges between two stages

    '''
    midpath = Path(midpath)
    newEdges = {}
    for i in range(len(edges)):
        newEdges[str(i)] = edges[i]
    #adata.uns['edges'] = newEdges
    f = open(midpath / str(iteration) / 'edges.txt','w')
    f.write(str(newEdges))
    f.close()
    return newEdges
def reupdateAttributes(adata, stage, results):
    '''
    update gaussian and gamma rep, top 100 differential genes, cell types of clusters to anndata

    parameters
    -------------------
    adata: anndata 
        The single-cell data
    stage: int
        The selected stage

    results: list
        A list contained top differential genes and cell types of clusters
    
    return
    -------------------
    adata: anndata
        updated anndata of input single-cell data
    '''
    stageids = adata.obs[adata.obs['stage'] == stage].index.tolist()
    tempadata = adata[stageids]
    adata.uns['rep'][str(stage)] = results[0]
    adata.uns['topGene'][str(stage)] = results[1]
    adata.uns['clusterType'][str(stage)] = results[2]
    return adata
