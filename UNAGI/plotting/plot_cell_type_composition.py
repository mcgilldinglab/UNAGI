import pickle
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc

def cell_type_composition(adata, cell_type_key, stage_key,ax=None,dpi=300,show_cutoff = 0.04, colormaps='Spectral',category_colors=None, save=None):
    '''
    Plot the cell type composition of each stage

    Parameters
    ----------
    adata : AnnData object
        Annotated data matrix.
    cell_type_key : str
        Key for cell type column in adata.obs.
    stage_key : str
        Key for stage column in adata.obs.
    ax : matplotlib axis, optional
        The default is None.
    dpi : int, optional
        The default is 300.
    show_cutoff : float, optional
        The default is 0.04.
    colormaps : str, optional
        The default is 'Spectral'.
    category_colors : list, optional
        The default is None.
    save : str, optional
        Path to save the figure. The default is None.
    '''

    all_types = adata.obs[cell_type_key].unique().tolist()
    all_types = sorted(all_types)
    stage_keys = adata.obs[stage_key].unique().tolist()
    stage_keys = sorted(stage_keys)
    stage_keys = stage_keys[::-1]
    stage_types = {key :[] for key in stage_keys}
    total_stage = len(stage_keys)
    for i in stage_keys:
        stage_adata = adata[adata.obs[stage_key] == i]
        stage_cells = len(stage_adata)
        for each in all_types:
            stage_types[i].append(len(stage_adata[stage_adata.obs[cell_type_key] == each])/stage_cells)
    multiplier = 0
    width = 0.3
    if ax is None:
        fig,ax = plt.subplots(figsize=(15, 3),dpi=dpi)

    #transform the type of list to array
    # y_pos = [0, 1]
    for i in stage_types.keys():
        stage_types[i] = np.array(stage_types[i])
    bottom = 0
    data = np.array(list(stage_types.values()))
    data_cum = data.cumsum(axis=1)
    if category_colors is None:
        category_colors = mpl.colormaps[colormaps](np.linspace(0.1, 1, data.shape[1]))
    for j, (types,c) in enumerate(zip(all_types,category_colors)):
        widths = data[:,j]
        starts = data_cum[:, j] - widths 
        bc = ax.barh(list(stage_types.keys()), widths, left = starts, label=types,color=c,height=0.8)
        labels = ["" if v > show_cutoff else "" for v in widths]  
        ax.bar_label(bc, labels=labels,label_type='center',fmt='%.2f',fontsize=9,color='black')
    ax.legend(ncol=len(all_types)//2+1, bbox_to_anchor=(0.1, 1.3),
                loc='upper left', fontsize='x-small')
    # threshold = 0.15
    for c in ax.containers:#comments out to disable percentage on the plot
        #v format %.2f
        labels = [f'%.2f%%'%(v*100) if v > show_cutoff else "" for v in c.datavalues] #comments out to disable percentage on the plot
        # labels = [v if v > 0.05 else "" for v in c.datavalues]  
        ax.bar_label(c, labels=labels, label_type="center",fmt='%.2f')#comments out to disable percentage on the plot
        # ax.bar_label(bc, labels=weight_count,label_type='center',fontsize=9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.grid(False) 
    if save is not None:
        plt.savefig(save)
    else:
        plt.show()