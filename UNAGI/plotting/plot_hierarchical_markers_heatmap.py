import scanpy as sc


def hierarchical_static_markers_heatmap(adata,stage,cluster,level,n_genes,stage_key,celltype_key,protein_encoding_gene=True,min_logfoldchange=1,dpi=None,save=None):
    '''
    Plot the heatmap of the hierarchical static markers of the chosen cluster and its siblings.

    Parameters
    ----------
    adata : AnnData object
        Annotated data matrix.
    stage : int
        Time point of the data.
    cluster : int
        Cluster id.
    level : int
        Level of the cluster.
    n_genes : int
        Number of genes to plot.
    stage_key : str
        Key for stage column in adata.obs.
    celltype_key : str
        Key for cell type column in adata.obs.
    protein_encoding_gene : bool, optional
        Whether to plot only protein encoding genes. The default is True.
    min_logfoldchange : int, optional
        Minimum log fold change. The default is 1.
    dpi : int, optional
        The default is None.
    save : str, optional
        Path to save the figure. The default is None.

    Returns
    ----------------
    None
    '''
    import warnings
    warnings.filterwarnings('ignore')
    if protein_encoding_gene:
        protein_coding_genes = []
        for each in adata.var.index:
            if each[:2] != 'AC' and each[:2] != 'AL' and each[:2] != 'AP' and each[:4] != 'LINC' and '.' not in each:
                protein_coding_genes.append(each)
        adata = adata[:,protein_coding_genes]
    if dpi is None:
        sc.set_figure_params(scanpy=True, dpi=80,dpi_save=300)
    else:
        sc.set_figure_params(scanpy=True, dpi=80,dpi_save=dpi)
    adata.obs[stage_key] = adata.obs[stage_key].astype(str)
    adata = adata[adata.obs[stage_key] == str(stage)]
    temp_clustertype = {}
    choice = adata.uns['hcmarkers'][str(stage)]['markers'][str(cluster)][str(level)]['ref'][0]
    scope = adata.uns['hcmarkers'][str(stage)]['markers'][str(cluster)][str(level)]['ref'][1]
    scope.remove(choice[0])
    adata.obs['leiden'] = adata.obs['leiden'].astype(int)
    choice_adata = adata[adata.obs['leiden'].isin(choice)]
    scope_adata = adata[adata.obs['leiden'].isin(scope)]
    choice_adata.obs['subset'] = 'chosen'
    scope_adata.obs['subset'] = 'siblings'
    apat = choice_adata.concatenate(scope_adata)
    apat.obs['subset'] = apat.obs['subset'].astype('category')
    apat.layers['scaled'] = sc.pp.scale(apat, copy=True).X
    marker_genes = []
    pos = []
    for i in range(len(temp_clustertype.keys())):
        temp = [i*10,i*10+9]
        pos.append(tuple(temp))
    sc.tl.rank_genes_groups(apat,'subset', method='wilcoxon',  n_genes=n_genes)
    apat.obs['leiden'] = apat.obs['leiden'].astype(str)
    marker_genes = apat.uns['rank_genes_groups']['names']['chosen']
    if save is not None:
        sc.pl.heatmap(apat, marker_genes, groupby=['subset',celltype_key,'leiden'],layer='scaled',swap_axes=True,vmin=-0.5, vmax=0.5, cmap='RdBu_r',figsize=(20, 16),save = save)
    else:
        # sc.pl.heatmap(apat, marker_genes, groupby=['subset',celltype_key,'leiden'],layer='scaled',swap_axes=True,vmin=-0.5, vmax=0.5, cmap='RdBu_r',figsize=(20, 16))
        sc.pl.rank_genes_groups_heatmap(apat,groupby=['subset',celltype_key,'leiden'],layer='scaled', min_logfoldchange=min_logfoldchange,n_genes=n_genes,swap_axes=True, use_raw=False,show_gene_labels=True,vmin=-0.5, vmax=0.5, cmap='RdBu_r',var_group_positions=pos,dendrogram=False,var_group_rotation=0,figsize=(20, 16))

if __name__ == '__main__':
    import pickle
    adata = sc.read_h5ad('small_1/dataset.h5ad')
    adata.uns = pickle.load(open('small_1/attribute.pkl', 'rb'))
    hierarchical_static_markers_heatmap(adata ,stage=0,choice=3,level=1,n_genes=20, stage_key = 'stage', celltype_key='ident', protein_encoding_gene=True, min_logfoldchange=1,dpi=100, save='test.png')