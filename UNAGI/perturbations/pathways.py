import pandas as pd
def get_top_pathways(adata, intensity, top_n=None, cutoff=None):
    """
    Get top pathways predictions after pathway perturbations at a given intensity.

    Parameters
    ----------
    adata : AnnData object
        Annotated data matrix.
    intensity : int
        Pertubration intensity.
    top_n : int, optional
        Number of top pathways to return. The default is None.
    cutoff : float, optional
        P-value cutoff. The default is None.
    """


    if top_n is not None:
        if 'top_compounds' not in adata.uns['pathway_perturbation_score'][str(intensity)]['total'].keys():
            print('All pertubred pathways are not statistically significant!')
            print('Here are the top %s pathways that are not statistically significant:'%(str(top_n)))
            temp = pd.DataFrame.from_dict(adata.uns['pathway_perturbation_score'][str(intensity)]['total']['down_compounds'])
            temp.rename(columns={'compound': 'pathways'}, inplace=True)
            temp.rename(columns={'drug_regulation': 'regulated genes'}, inplace=True)
            return temp[:top_n]
        else:
            temp = pd.DataFrame.from_dict(adata.uns['pathway_perturbation_score'][str(intensity)]['total']['top_compounds'])
            temp.rename(columns={'compound': 'pathways'}, inplace=True)
            temp.rename(columns={'drug_regulation': 'regulated genes'}, inplace=True)
            return temp[:top_n]
    elif cutoff is not None:
        if 'top_compounds' not in adata.uns['pathway_perturbation_score'][str(intensity)]['total'].keys():
            print('All pertubred pathways are not statistically significant!')
            print('Here are the pathways that are not statistically significant:')
            temp = pd.DataFrame.from_dict(adata.uns['pathway_perturbation_score'][str(intensity)]['total']['down_compounds'])
            temp.rename(columns={'compound': 'pathways'}, inplace=True)
            temp.rename(columns={'drug_regulation': 'regulated genes'}, inplace=True)
            return temp
        else:
            temp = pd.DataFrame.from_dict(adata.uns['pathway_perturbation_score'][str(intensity)]['total']['top_compounds'])
            temp.rename(columns={'compound': 'pathways'}, inplace=True)
            temp.rename(columns={'drug_regulation': 'regulated genes'}, inplace=True)
            return temp[temp['pval_adjusted'] < cutoff]
        
    else:
        print('Please specify top_n or cutoff')