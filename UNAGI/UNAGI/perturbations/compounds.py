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
        return pd.DataFrame.from_dict(adata.uns['drug_perturbation_score'][str(intensity)]['total']['top_compounds'])[:top_n]
    elif cutoff is not None:
        temp = pd.DataFrame.from_dict(adata.uns['drug_perturbation_score'][str(intensity)]['total']['top_compounds'])
        return temp[temp['pval_adjusted'] < cutoff]
    else:
        print('Please specify top_n or cutoff')