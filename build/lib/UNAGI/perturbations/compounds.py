import pandas as pd
def get_top_compounds(adata, intensity, top_n=None, cutoff=None):
    '''
    Get top compounds predictions after compound perturbations at a given intensity.

    Parameters
    ----------
    adata : AnnData object
        Annotated data matrix.
    intensity : int
        Pertubration intensity.
    top_n : int, optional
        Number of top compounds to return. The default is None.
    cutoff : float, optional
        P-value cutoff. The default is None.
    '''
    if top_n is not None:
        if 'top_compounds'not in adata.uns['drug_perturbation_score'][str(intensity)]['total'].keys():
            if 'down_compounds' in adata.uns['drug_perturbation_score'][str(intensity)]['total'].keys():
                print('All pertubred compounds are not statistically significant!')
                print('Here are the top %s compounds that are not statistically significant:'%(str(top_n)))
                return pd.DataFrame.from_dict(adata.uns['drug_perturbation_score'][str(intensity)]['total']['down_compounds'])[:top_n]
        return pd.DataFrame.from_dict(adata.uns['drug_perturbation_score'][str(intensity)]['total']['top_compounds'])[:top_n]
    elif cutoff is not None:
        if 'top_compounds'not in adata.uns['drug_perturbation_score'][str(intensity)]['total'].keys():
            if 'down_compounds' in adata.uns['drug_perturbation_score'][str(intensity)]['total'].keys():
                print('All pertubred compounds are not statistically significant!')
                print('Here are the compounds that are not statistically significant:')
                return pd.DataFrame.from_dict(adata.uns['drug_perturbation_score'][str(intensity)]['total']['down_compounds'])
        temp = pd.DataFrame.from_dict(adata.uns['drug_perturbation_score'][str(intensity)]['total']['top_compounds'])
        return temp[temp['pval_adjusted'] < cutoff]
    else:
        print('Please specify top_n or cutoff')