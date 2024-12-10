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
        if 'overall' not in adata.uns['drug_perturbation_score'][str(intensity)].keys():
            dict_results = {}
            for each_track in adata.uns['drug_perturbation_score'][str(intensity)].keys():
                if 'top_compounds' not in adata.uns['drug_perturbation_score'][str(intensity)][each_track].keys():
                    print('All pertubred compounds are not statistically significant in track %s!'%(each_track))
                    dict_results[each_track] = {}
                else:
                    dict_results[each_track] = pd.DataFrame.from_dict(adata.uns['drug_perturbation_score'][str(intensity)][each_track]['top_compounds'])[:top_n]
            print('You are checking the perturbation results for individual tracks, the returning results are stored in the dictionary.\n You can access the results by using the track name as the key.')
            print('Here are the keys for the dictionary:')
            for key in dict_results.keys():
                print(key)
            return dict_results
        else:
            if 'top_compounds' not in adata.uns['drug_perturbation_score'][str(intensity)]['overall'].keys():
                if 'down_compounds' in adata.uns['drug_perturbation_score'][str(intensity)]['overall'].keys():
                    print('All pertubred compounds are not statistically significant!')
                    print('Here are some top insiginifcant compounds that are not statistically significant:')
                    return pd.DataFrame.from_dict(adata.uns['drug_perturbation_score'][str(intensity)]['overall']['down_compounds'])[:top_n]
            return pd.DataFrame.from_dict(adata.uns['drug_perturbation_score'][str(intensity)]['overall']['top_compounds'])[:top_n]
    elif cutoff is not None:
        if 'overall' not in adata.uns['drug_perturbation_score'][str(intensity)].keys():
            dict_results = {}
            for each_track in adata.uns['drug_perturbation_score'][str(intensity)].keys():
                if 'top_compounds' not in adata.uns['drug_perturbation_score'][str(intensity)][each_track].keys():
                    print('All pertubred compounds are not statistically significant in track %s!'%(each_track))
                    dict_results[each_track] = {}
                else:
                    temp = pd.DataFrame.from_dict(adata.uns['drug_perturbation_score'][str(intensity)][each_track]['top_compounds'])
                    dict_results[each_track] = temp[temp['pval_adjusted'] < cutoff]
            print('You are checking the perturbation results for individual tracks, the returning results are stored in the dictionary.\n You can access the results by using the track name as the key.')
            print('Here are the keys for the dictionary:')
            for key in dict_results.keys():
                print(key)
            return dict_results
        else:
            if 'top_compounds' not in adata.uns['drug_perturbation_score'][str(intensity)]['overall'].keys():
                if 'down_compounds' in adata.uns['drug_perturbation_score'][str(intensity)]['overall'].keys():
                    print('All pertubred compounds are not statistically significant!')
                    print('Here are some top insiginifcant compounds that are not statistically significant:')
                    return pd.DataFrame.from_dict(adata.uns['drug_perturbation_score'][str(intensity)]['overall']['down_compounds'])
            temp = pd.DataFrame.from_dict(adata.uns['drug_perturbation_score'][str(intensity)]['overall']['top_compounds'])
            return temp[temp['pval_adjusted'] < cutoff]
    else:
        print('Please specify top_n or cutoff')