import scanpy as sc
import pickle
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
def plot_hc_dendrogram(adata,stage_key,celltype_key,save=False,dpi=None):
    '''
    Plot the dendrogram of the hierarchical static markers of each stage.

    Parameters
    ----------
    adata : AnnData object
        Annotated data matrix.
    stage_key : str
        Key for stage column in adata.obs.
    celltype_key : str
        Key for cell type column in adata.obs.
    save : bool, optional
        Whether to save the figure. The default is False.
    dpi : int, optional
        The default is None.

    Returns
    -------------
    None
    '''
    for stage in list(adata.obs[stage_key].unique()):
        sch.dendrogram(adata.uns['hcmarkers'][str(stage)]['Z'],no_plot=True)
        #replace the labels of the leaves with the labels of the original data
        leaves = sch.leaves_list(adata.uns['hcmarkers'][str(stage)]['Z'])
        adata.obs['leiden_celltype']  = adata.obs['leiden'].astype(str) +'_'+ adata.obs[celltype_key].astype(str)
        stage0 = adata[adata.obs[stage_key] == str(stage)]
        # sc.pl.umap(stage0,color=celltype_key,show=False)
        new_leaves = []
        stage0.obs['leiden'] == stage0.obs['leiden'].astype(str)
        for each in leaves:
            temp = stage0[stage0.obs['leiden'] == str(each)]
            new_leaves.append(temp.obs['leiden_celltype'].unique()[0])
        if dpi is None:
            plt.figure(figsize=(10,10),dpi=100)
        else:
            plt.figure(figsize=(10,10),dpi=dpi)
        ax = plt.gca()
        sch.dendrogram(adata.uns['hcmarkers'][str(stage)]['Z'],ax=ax)
        ax.set_xticklabels(new_leaves)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.title(stage)
        if save:
            plt.savefig('dendrogram_stage_%d.pdf'%(stage))
        plt.show()
if __name__ == '__main__':
    adata = sc.read_h5ad('small_1/dataset.h5ad')
    adata.uns = pickle.load(open('small_1/attribute.pkl','rb'))
    plot_hc_dendrogram(adata,'stage','ident')
