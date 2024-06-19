from typing import Optional, Sequence, Dict, Any
import scipy.cluster.hierarchy as sch
import scanpy as sc
import numpy as np
from pandas.api.types import CategoricalDtype
import pandas as pd
from scanpy.tools._utils import _choose_representation
from anndata import AnnData
from ..dynamic_graphs.distDistance import fitClusterGaussianRepresentation,calculateKL

def getclusterreps(data):
    '''
    Get the sampled Norm distribution of Z space for each cluster

    Parameters
    ----------
    data: pandas dataframe
        The hidden representation of Z space. The index of the dataframe is the cluster id. The columns of the dataframe are the hidden representation of Z space.

    Returns
    -------
    outs: dictionary
        The sampled Norm distribution of Z space. The keys of the dictionary are the cluster id. The values of the dictionary are the sampled Norm distribution of Z space.

    '''
    clusterid = data.index.unique().tolist()
    out = {}
    for each in clusterid:
        out[each] = fitClusterGaussianRepresentation(data.loc[each].T.values)
    return out
def calculateDistance(reps):
    '''
    Calculate the distance between the sampled Norm distribution of Z space

    Parameters
    ----------
    reps: dictionary
        The sampled Norm distribution of Z space. The keys of the dictionary are the cluster id. The values of the dictionary are the sampled Norm distribution of Z space.

    Returns
    -------
    df: pandas dataframe
        The distance between the sampled Norm distribution of Z space. The index of the dataframe is the cluster id. The columns of the dataframe are the cluster id.
    '''
    outs = {}
    for i, each1 in enumerate(list(reps.keys())):
        outs[each1] = []
        for j, each2 in enumerate(list(reps.keys())):
            outs[each1].append(((calculateKL(reps[each1], reps[each2])+calculateKL(reps[each2], reps[each1]))/2))
    df = pd.DataFrame(outs,index=list(reps.keys()))
    df.columns = list(reps.keys())
    df.index = df.index.astype(int)
    df = df.sort_index()
    df.columns =  df.columns.astype(int)
    df = df.sort_index(axis=1)
    return df
def calculateDistanceUMAP(umaps):
    '''
    Calculate the distance between umap coordinates of Z space

    Parameters
    ----------
    umaps: pandas dataframe
        The umap coordinates of Z space. The index of the dataframe is the cluster id. The columns of the dataframe are the umap coordinates of Z space.

    Returns 
    -------
    df: pandas dataframe
        The distance between umap coordinates of Z space. The index of the dataframe is the cluster id. The columns of the dataframe are the cluster id.
    '''
    outs = {}

    for i, each1 in enumerate(list(umaps.index)):
        outs[each1] = []
        
        for j, each2 in enumerate(list(umaps.index)):
            outs[each1].append(np.linalg.norm(umaps.loc[each1]- umaps.loc[each2]))
    df = pd.DataFrame(outs,index=list(umaps.index))
    df.columns = list(umaps.index)
    df.index = df.index.astype(int)
    df = df.sort_index()
    df.columns =  df.columns.astype(int)
    df = df.sort_index(axis=1)
    return df
def mydendrogram(
    adata: AnnData,
    groupby: str,
    n_pcs: Optional[int] = None,
    use_rep: Optional[str] = None,
    var_names: Optional[Sequence[str]] = None,
    use_raw: Optional[bool] = None,
    cor_method: str = "pearson",
    linkage_method: str = "complete",
    optimal_ordering: bool = False,
    key_added: Optional[str] = None,
    inplace: bool = True,
) -> Optional[Dict[str, Any]]:
    """\
    Modified from scanpy.tl.dendrogram. 
    Computes a hierarchical clustering for the given `groupby` categories.

    By default, the latent representation 'Z' is used.

    Alternatively, a list of `var_names` (e.g. genes) can be given.

    Average values of either `var_names` or components are used
    to compute a correlation matrix.

    The hierarchical clustering can be visualized using
    :func:`scanpy.pl.dendrogram` or multiple other visualizations that can
    include a dendrogram: :func:`~scanpy.pl.matrixplot`,
    :func:`~scanpy.pl.heatmap`, :func:`~scanpy.pl.dotplot`,
    and :func:`~scanpy.pl.stacked_violin`.

    .. note::
        The computation of the hierarchical clustering is based on predefined
        groups and not per cell. The correlation matrix is computed using by
        default pearson but other methods are available.

    Parameters
    ----------
    adata
        Annotated data matrix
    {n_pcs}
    {use_rep}
    var_names
        List of var_names to use for computing the hierarchical clustering.
        If `var_names` is given, then `use_rep` and `n_pcs` is ignored.
    use_raw
        Only when `var_names` is not None.
        Use `raw` attribute of `adata` if present.
    cor_method
        correlation method to use.
        Options are 'pearson', 'kendall', and 'spearman'
    linkage_method
        linkage method to use. See :func:`scipy.cluster.hierarchy.linkage`
        for more information.
    optimal_ordering
        Same as the optimal_ordering argument of :func:`scipy.cluster.hierarchy.linkage`
        which reorders the linkage matrix so that the distance between successive
        leaves is minimal.
    key_added
        By default, the dendrogram information is added to
        `.uns[f'dendrogram_{{groupby}}']`.
        Notice that the `groupby` information is added to the dendrogram.
    inplace
        If `True`, adds dendrogram information to `adata.uns[key_added]`,
        else this function returns the information.

    Returns
    -------
    If `inplace=False`, returns dendrogram information,
    else `adata.uns[key_added]` is updated with it.
    """
    if isinstance(groupby, str):
        # if not a list, turn into a list
        groupby = [groupby]
    for group in groupby:
        if group not in adata.obs_keys():
            raise ValueError(
                "groupby has to be a valid observation. "
                f"Given value: {group}, valid observations: {adata.obs_keys()}"
            )
        if not isinstance(adata.obs[group].dtype, CategoricalDtype):
            raise ValueError(
                "groupby has to be a categorical observation. "
                f"Given value: {group}, Column type: {adata.obs[group].dtype}"
            )

    if var_names is None:
        rep_df = pd.DataFrame(
            _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)
        )
        categorical = adata.obs[groupby[0]]
        if len(groupby) > 1:
            for group in groupby[1:]:
                # create new category by merging the given groupby categories
                categorical = (
                    categorical.astype(str) + "_" + adata.obs[group].astype(str)
                ).astype("category")
        categorical.name = "_".join(groupby)

        rep_df.set_index(categorical, inplace=True)
        categories = rep_df.index.categories
    else:
        gene_names = adata.raw.var_names if use_raw else adata.var_names

    # aggregate values within categories using 'mean'
    
    mean_df = rep_df.groupby(level=0).mean()

    import scipy.cluster.hierarchy as sch
    from scipy.spatial import distance
    test = getclusterreps(rep_df)
    test2 = calculateDistance(test)
    corr_matrix = mean_df.T.corr(method=cor_method)
    corr_condensed = distance.squareform(test2)

    z_var = sch.linkage(
        corr_condensed, method=linkage_method, optimal_ordering=optimal_ordering
    )
    dendro_info = sch.dendrogram(z_var, labels=list(categories), no_plot=True)

    dat = dict(
        linkage=z_var,
        groupby=groupby,
        use_rep=use_rep,
        cor_method=cor_method,
        linkage_method=linkage_method,
        categories_ordered=dendro_info["ivl"],
        categories_idx_ordered=dendro_info["leaves"],
        dendrogram_info=dendro_info,
        correlation_matrix=corr_matrix.values,
    )

    if inplace:
        if key_added is None:
            key_added = f'dendrogram_{"_".join(groupby)}'
        # logg.info(f"Storing dendrogram info using `.uns[{key_added!r}]`")
        adata.uns[key_added] = dat
    else:
        return dat
def mydendrogramUMAP(
    adata: AnnData,
    groupby: str,
    n_pcs: Optional[int] = None,
    use_rep: Optional[str] = None,
    var_names: Optional[Sequence[str]] = None,
    use_raw: Optional[bool] = None,
    cor_method: str = "pearson",
    linkage_method: str = "complete",
    optimal_ordering: bool = False,
    key_added: Optional[str] = None,
    inplace: bool = True,
) -> Optional[Dict[str, Any]]:
    """\
    Modify from scanpy.tl.dendrogram but the distance is between umap coordinates of latent representation 'Z' space
    
    Computes a hierarchical clustering for the given `groupby` categories.


    Alternatively, a list of `var_names` (e.g. genes) can be given.

    Average values of either `var_names` or components are used
    to compute a correlation matrix.

    The hierarchical clustering can be visualized using
    :func:`scanpy.pl.dendrogram` or multiple other visualizations that can
    include a dendrogram: :func:`~scanpy.pl.matrixplot`,
    :func:`~scanpy.pl.heatmap`, :func:`~scanpy.pl.dotplot`,
    and :func:`~scanpy.pl.stacked_violin`.

    .. note::
        The computation of the hierarchical clustering is based on predefined
        groups and not per cell. The correlation matrix is computed using by
        default pearson but other methods are available.

    Parameters
    ----------
    adata
        Annotated data matrix
    {n_pcs}
    {use_rep}
    var_names
        List of var_names to use for computing the hierarchical clustering.
        If `var_names` is given, then `use_rep` and `n_pcs` is ignored.
    use_raw
        Only when `var_names` is not None.
        Use `raw` attribute of `adata` if present.
    cor_method
        correlation method to use.
        Options are 'pearson', 'kendall', and 'spearman'
    linkage_method
        linkage method to use. See :func:`scipy.cluster.hierarchy.linkage`
        for more information.
    optimal_ordering
        Same as the optimal_ordering argument of :func:`scipy.cluster.hierarchy.linkage`
        which reorders the linkage matrix so that the distance between successive
        leaves is minimal.
    key_added
        By default, the dendrogram information is added to
        `.uns[f'dendrogram_{{groupby}}']`.
        Notice that the `groupby` information is added to the dendrogram.
    inplace
        If `True`, adds dendrogram information to `adata.uns[key_added]`,
        else this function returns the information.

    Returns
    -------
    If `inplace=False`, returns dendrogram information,
    else `adata.uns[key_added]` is updated with it.
    """
    if isinstance(groupby, str):
        # if not a list, turn into a list
        groupby = [groupby]
    for group in groupby:
        if group not in adata.obs_keys():
            raise ValueError(
                "groupby has to be a valid observation. "
                f"Given value: {group}, valid observations: {adata.obs_keys()}"
            )
        if not isinstance(adata.obs[group].dtype, CategoricalDtype):
            raise ValueError(
                "groupby has to be a categorical observation. "
                f"Given value: {group}, Column type: {adata.obs[group].dtype}"
            )

    if var_names is None:
        rep_df = pd.DataFrame(
            _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)
        )
        categorical = adata.obs[groupby[0]]
        if len(groupby) > 1:
            for group in groupby[1:]:
                # create new category by merging the given groupby categories
                categorical = (
                    categorical.astype(str) + "_" + adata.obs[group].astype(str)
                ).astype("category")
        categorical.name = "_".join(groupby)

        rep_df.set_index(categorical, inplace=True)
        categories = rep_df.index.categories
    else:
        gene_names = adata.raw.var_names if use_raw else adata.var_names

    # aggregate values within categories using 'mean'
    
    mean_df = rep_df.groupby(level=0).mean()

    import scipy.cluster.hierarchy as sch
    from scipy.spatial import distance
    # test = getclusterrepsUMAP(rep_df)

    test2 = calculateDistanceUMAP(mean_df)
    # corr_matrix = mean_df.T.corr(method=cor_method)
    corr_condensed = distance.squareform(test2)

    z_var = sch.linkage(
        corr_condensed, method=linkage_method, optimal_ordering=optimal_ordering
    )
    dendro_info = sch.dendrogram(z_var, labels=list(categories), no_plot=True)

    dat = dict(
        linkage=z_var,
        groupby=groupby,
        use_rep=use_rep,
        cor_method=cor_method,
        linkage_method=linkage_method,
        categories_ordered=dendro_info["ivl"],
        categories_idx_ordered=dendro_info["leaves"],
        dendrogram_info=dendro_info,
        # correlation_matrix=corr_matrix.values,
    )

    if inplace:
        if key_added is None:
            key_added = f'dendrogram_{"_".join(groupby)}'
        # logg.info(f"Storing dendrogram info using `.uns[{key_added!r}]`")
        adata.uns[key_added] = dat
    else:
        return dat
def get_siblings_at_each_level(leaf_id, Z):
    """
    Get the sibling clusters for the given leaf at each level of the hierarchical clustering.

    Parameters
    ----------  
    leaf_id: int
        The leaf id of the given leaf.
    Z: numpy.ndarray
        The linkage matrix of the hierarchical clustering.

    Returns
    -------
    siblings: list
        The sibling clusters for the given leaf at each level of the hierarchical clustering.
    """
    n = Z.shape[0] + 1  # The total number of initial leaves
    clusters = {i: [i] for i in range(n)}  # Initialize clusters with single leaves
    siblings = []

    # Helper function to find the parent node for the given leaf or cluster
    def find_parent(node):
        for i, (left, right, _, _) in enumerate(Z):
            if node in [left, right]:
                return i + n, left if node != left else right
        return None, None

    # Build clusters and collect siblings at different levels
    for k in range(n - 1):
        left, right = int(Z[k, 0]), int(Z[k, 1])
        clusters[n + k] = clusters[left] + clusters[right]

    # Trace back from the leaf to the root and collect siblings
    current_node = leaf_id
    while True:
        parent, sibling_node = find_parent(current_node)
        if parent is None:  # Reached the root
            break
        # Check if sibling_node is a leaf or a cluster and collect all leaves
        sibling_leaves = clusters[sibling_node]
        siblings.append(sibling_leaves)
        current_node = parent

    return siblings[::-1]  # Reverse to have siblings from bottom to top
def map_leaves_to_label(categories_idx_ordered, categories_ordered):
    table = {}
    for i, each in enumerate(categories_idx_ordered):
        table[each] = int(categories_ordered[i])
    return table
def my_logfold_change(adata1, adata2,log_fold_change_cutoff=None,abs = False):
    '''
    Define the log fold change between two datasets. (data is already log transformed) LogFoldChange = Log(mean2) - Log(mean1)

    Parameters
    ----------
    adata1: AnnData
        The annotated data matrix.
    adata2: AnnData
        The annotated data matrix.
    log_fold_change_cutoff: float   
        The cutoff of the log fold change to select the genes. Default is None.
    abs: bool
        If True, the absolute value of the log fold change will be used. Default is False.

    Returns
    -------
    df: pandas dataframe
        The log fold change between two datasets. The columns of the dataframe are the log fold change, the names of the genes, the mean of the genes in the first dataset and the mean of the genes in the second dataset.

    '''
    genenames = adata1.var_names.tolist()
    mean1 = np.mean(adata1.X.toarray(),axis=0)
    mean2 = np.mean(adata2.X.toarray(),axis=0)
    logfold_change = mean2 - mean1
    logfold_change_dict = {}
    order = np.argsort(logfold_change)[::-1]
    mean2 = mean2[order]
    mean1 = mean1[order]
    for i, each in enumerate(genenames):
        logfold_change_dict[each] = logfold_change[i]
    if abs ==True:
        temp = {k: v for k, v in sorted(logfold_change_dict.items(), key=lambda item: np.abs(item[1]),reverse=True)}
    else:
        temp = {k: v for k, v in sorted(logfold_change_dict.items(), key=lambda item: item[1],reverse=True)}

    df = pd.DataFrame()
    df['log_fold_changes'] = temp.values()
    df['names'] = temp.keys()

    if log_fold_change_cutoff is not None:
        df = df[np.abs(df['log_fold_changes'])>log_fold_change_cutoff]
    df['Mean_compared_clusters'] = mean1
    df['Mean_selected_clusters'] = mean2
    return df
def build_reference_siblings(dendrogram):
    '''
    Find the sibling clusters for each cluster at each level of the hierarchical clustering.

    Parameters
    ----------
    dendrogram: dict
        The data structure to store the hierarchical clustering information. The data structure is the output of the function mydendrogram.
    
    Returns
    -------
    out: dict
        The data structure to store the sibling clusters for each cluster at each level of the hierarchical clustering. The keys of the dictionary are the cluster ids. The values of the dictionary are the sibling clusters at each level of the hierarchical clustering. 
    '''
    total_leaves = len(dendrogram['dendrogram_info']['ivl'])
    table = map_leaves_to_label(dendrogram['categories_idx_ordered'], dendrogram['categories_ordered'])
    out = {}
    for leave_id in range(total_leaves):
        siblings = get_siblings_at_each_level(leave_id, dendrogram['linkage'])
        leaf_node_id = table[leave_id]
        out[str(leaf_node_id)] = {}
        level_table = {}
        memory = []
        for i, sibs in enumerate(siblings[::-1]):
            translated_sibs = [table[each] for each in sibs]
            memory+=translated_sibs
            level_table[len(siblings)-1-i] = memory.copy()
        for level, sibs in enumerate(siblings):
            out[str(leaf_node_id)][str(level)] = level_table[level]
    return out
def gethcmarkers(adata, selected,groups,cutoff=0.05):
    '''
    Get the hierarchical clustering markers for the selected cluster and the background clusters.

    Parameters
    ----------
    adata: AnnData
        The annotated data matrix.
    selected: int
        The cluster id of the selected cluster.
    groups: list    
        The cluster ids of the background clusters.
    cutoff: float
        The cutoff of the adjusted p-value to select the hierarchical clustering markers.

    Returns 
    -------
    positive_marker: pandas dataframe
        The positive hierarchical clustering markers for the selected cluster. 
    negative_marker: pandas dataframe
        The negative hierarchical clustering markers for the selected cluster.
    '''
    ref = [[selected],[selected]+groups]
    ids = []
    ids = adata[adata.obs['leiden'] == str(selected)].obs.index.tolist()
    selectedids = ids.copy()
    groupids = []
    for each in groups:
        groupids +=adata[adata.obs['leiden'] == str(each)].obs.index.tolist()
    ids += groupids.copy()
    new = adata[ids]
    new.obs['compare'] = None
    new.obs.loc[new.obs.index.isin(selectedids), "compare"] = 'selected'
    new.obs.loc[new.obs.index.isin(groupids), "compare"] = 'background'
    sc.tl.rank_genes_groups(new,groupby='compare',method='wilcoxon',rankby_abs=False)
    selected = new[selectedids]
    background = new[groupids]
    df1 = my_logfold_change(background,selected)
    df1 = df1.set_index('names')
    df = pd.DataFrame.from_dict({key: new.uns['rank_genes_groups'][key]['selected'] for key in['names','pvals_adj','scores']})
    df = df.set_index('names')
    df = df.join(df1)
    positive_marker = df[df['log_fold_changes']>0]
    negative_marker = df[df['log_fold_changes']>0]
    positive_marker= positive_marker[positive_marker['pvals_adj'] <cutoff]
    negative_marker= negative_marker[negative_marker['pvals_adj'] <cutoff]
    
    return positive_marker, negative_marker,ref
def get_stage_hcmarkers(adata,key,use_rep,cutoff=0.05):
    '''
    Perform hierarchical clustering on the stage and get the hierarchical clustering markers for each cluster.

    Parameters
    ----------
    adata: AnnData
        The annotated data matrix.
    key: str
        The key of the cluster information in adata.obs.
    use_rep: str
        The key of the representation in adata.obsm.
    cutoff: float
        The cutoff of the adjusted p-value to select the hierarchical clustering markers.

    Returns
    -------
    Z: numpy.ndarray
        The linkage matrix of the hierarchical clustering.
    order: list
        The order of the clusters in the hierarchical clustering.
    out_table: dict
        The data structure to store the hierarchical clustering markers for each cluster.
    '''
    if use_rep == 'z':
        mydendrogram(adata,use_rep='z',groupby=key)
    elif use_rep == 'umaps':
        mydendrogramUMAP(adata,use_rep='X_umap',groupby=key)
    
    test = build_reference_siblings(adata.uns['dendrogram_'+key])
    Z = adata.uns['dendrogram_'+key]['linkage']
    order = adata.uns['dendrogram_'+key]['categories_ordered']
    out_table = {}
    for each in list(test.keys()):
        
        out_table[each] = {}
        for level in test[each]:
            out_table[each][level] = {}
            out_table[each][level]['chosen'] = {}
            pos_markers, neg_markers, ref = gethcmarkers(adata,int(each),test[each][level],cutoff=cutoff)
            out_table[each][level]['chosen']['pos'] = pos_markers
            out_table[each][level]['chosen']['neg'] = neg_markers
            out_table[each][level]['ref'] = ref
    return Z,order,out_table
def get_dataset_hcmarkers(adata,stage_key,cluster_key,use_rep,cutoff=0.05):
    '''
    Perform hierarchical clustering on the dataset and get the hierarchical clustering markers for each stage.

    Parameters
    ----------
    adata: AnnData
        The annotated data matrix.
    stage_key: str
        The key of the stage information in adata.obs.
    cluster_key: str
        The key of the cluster information in adata.obs.
    use_rep: str
        The key of the representation in adata.obsm.
    cutoff: float
        The cutoff of the adjusted p-value to select the hierarchical clustering markers. Default is 0.05.

    Returns 
    -------
    hcmarkers: dict
        The hierarchical clustering markers for each stage. The keys of the dictionary are the cluster information in adata.obs. The values of the dictionary are the positive and negative hierarchical clustering markers for each cluster. The positive and negative hierarchical clustering markers are pandas dataframe with the columns of the names of the genes, the log fold change and the adjusted p-value.
    '''
    hcmarkers = {}
    for each in list(adata.obs[stage_key].unique()):
        hcmarkers[str(each)] = {}
        stage_adata = adata[adata.obs[stage_key] == each]
        Z,order,markers = get_stage_hcmarkers(stage_adata,cluster_key,use_rep=use_rep,cutoff=cutoff)
        hcmarkers[str(each)]['Z'] = Z
        hcmarkers[str(each)]['order'] = order
        hcmarkers[str(each)]['markers'] = markers
    return hcmarkers
# get_dataset_hcmarkers(adata,'stage','leiden')
#unit test
if __name__ == '__main__':
    import pickle
    # adata = sc.read_h5ad('/mnt/md0/yumin/UNAGI_old/data/mes/4/org_test/dataset.h5ad')
    adata = sc.read('/mnt/md0/yumin/UNAGI_pyro/src/small_0/dataset.h5ad')
    # adata.uns = pickle.load(open('/mnt/md0/yumin/UNAGI_old/data/mes/4/org_test/attribute.pkl', 'rb'))
    adata.uns = pickle.load(open('/mnt/md0/yumin/UNAGI_pyro/src/small_0/attribute.pkl', 'rb'))
    
    adata.uns['hcmarkers'] = get_dataset_hcmarkers(adata,stage_key='stage',cluster_key='leiden',use_rep='umaps')
    # with open('/mnt/md0/yumin/UNAGI_old/data/mes/4/org_test/attribute_1109_newhc.pkl','wb') as f:
    #     pickle.dump(adata.uns,f)