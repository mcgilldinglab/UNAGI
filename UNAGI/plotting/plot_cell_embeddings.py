
# colodict = {}
# for each in zip(all_types,category_colors):
#     colodict[each[0]] = rgb2hex(each[1])
import scanpy as sc
import gc

import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
from sklearn.neighbors import kneighbors_graph

def plot_with_colormap(values,color_dict):
    '''
    The color scheme the cell types are plotted with.
    
    Parameters
    ----------
    values : list
        List of cell types.
    color_dict : dict
        Dictionary of cell types and their colors.
    
    Returns
    -------
    color_dict : dict
        Dictionary of cell types and their colors.
    '''
    color_list = [[0.36862745, 0.30980392, 0.63529412, 1.        ],'tab:pink','tab:olive','tab:cyan','gold', 'springgreen','coral','skyblue','tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','yellow','aqua', 'turquoise','orangered', 'lightblue','darkorchid', 'fuchsia','royalblue','slategray', 'silver', 'teal', 'fuchsia','grey','indigo','khaki','magenta','tab:gray']
    # random.shuffle(color_list)
    values = list(set(values))
    values = sorted(values)
    for i, value in enumerate(values):
        if value not in list(color_dict.keys()):
            color_dict[value] = color_list[(len(list(color_dict.keys()))+1)]
    return color_dict
def plot_stages_latent_representation(adatas, cell_type_key, stage_key,color_scheme=None,ax=None,dpi=300,save=None):
    '''
    Plot the latent representation of the cells colored by cell type and leiden clusters.

    Parameters
    ----------
    adatas : AnnData object
        Annotated data matrix.
    cell_type_key : str
        Key for cell type column in adata.obs.
    stage_key : str
        Key for stage column in adata.obs.
    color_scheme : dict, optional
        Dictionary of cell types and their colors. The default is None.
    ax : matplotlib axis, optional  
        The default is None.
    dpi : int, optional
        The default is 300.
    save : str, optional
        Path to save the figure. The default is None.

    Returns
    --------------

    '''
    sc.set_figure_params(scanpy=True, dpi=dpi)
    consistency = []
    ariss= []
    NMIs = []
    silhouettes = []
    # ITERATION= 5
    stage_keys = adatas.obs[stage_key].unique().tolist()
    stage_keys = sorted(stage_keys)
    stage_keys = stage_keys[::-1]

    if color_scheme is None:
        color_dict_unagi = {}
    else:
        color_dict_unagi = color_scheme

    color_dict_leiden = {}
    color_dict_groundtruth = {}
    total_adata = 0
    count=0
    NMI = 0
    silhouettes =0
    aris = 0

    fig, ax = plt.subplots(4,2, figsize=(10,15))
    for i,stage in enumerate(stage_keys):
        
        temp_count = 0
        #check the type of adatas.obs[stage_key]
        if adatas.obs[stage_key].dtype == 'str':
            stage = str(stage)
        elif adatas.obs[stage_key].dtype == 'int':
            stage = int(stage)
        print(len(adatas.obs[adatas.obs[stage_key] == stage].index.tolist()))

        adata = adatas[adatas.obs[adatas.obs[stage_key] == stage].index.tolist()]
    #         print(len(adata))
        adata.obs['UNAGI'] = adata.obs[cell_type_key].astype('category')
        
        # adata.obs['Ground Truth'] = adata.obs['name.simple'].astype('category')
    
        adata.obs['leiden'] = adata.obs['leiden'].astype('category')

        sorted_list = sorted(list(adata.obs['UNAGI'].unique()))
        color_dict_unagi = plot_with_colormap(sorted_list,color_dict_unagi)
        adata.obs['leiden'] = adata.obs['leiden'].astype('string')
        sc.pl.umap(adata,color='UNAGI',ax=ax[i,0], show=False,palette=color_dict_unagi,title=str(stage_keys[i]))
        sc.pl.umap(adata,color='leiden',ax=ax[i,1], show=False,title = str(stage_keys[i]))
        total_adata+=len(adata)
        count+=temp_count
        temp_ari = adjusted_rand_score(adata.obs['name.simple'],adata.obs['UNAGI'] )
        temp_nmi = normalized_mutual_info_score(adata.obs['name.simple'],adata.obs['UNAGI'])
        temp_silhouette_score = silhouette_score(adata.obsm['z'], adata.obs['leiden'])
        print('ARI: ', temp_ari)
        print('NMIs: ', temp_nmi)
        print('silhouette score: ', temp_silhouette_score)
        NMI += temp_nmi
        silhouettes += temp_silhouette_score
        aris += temp_ari
    consistency.append(count/total_adata)
    ariss.append(aris/4)
    NMIs.append(NMI/4)
    plt.tight_layout()
    if save is not None:
        plt.savefig(save,dpi=dpi)
    else:
        plt.show()
    print('ARIs: ', ariss)
    print('NMI: ', NMIs)
    print('silhouette score: ', silhouettes/4)
# if __name__ == '__main__':
#     adata = sc.read_h5ad('../dataset.h5ad')
#     plot_stages_latent_representation(adata,'ident','stage')