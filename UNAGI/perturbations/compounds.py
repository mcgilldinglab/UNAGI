import pandas as pd
def get_top_compounds(adata, intensity, top_n=None, cutoff=None,selected_track=None):
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
    selected_track: str, optional
        Show the resultsof track `selected_track`, must run perturbation for the track! The default is None: show overall results.
    '''
    if top_n is not None:
        if 'overall' not in adata.uns['drug_perturbation_score'][str(intensity)].keys():
            dict_results = {}
            if selected_track == None:
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
                if selected_track not in adata.uns['drug_perturbation_score'][str(intensity)].keys():
                    raise ValueError('Not a valid track!')
                else:
                    temp = pd.DataFrame.from_dict(adata.uns['drug_perturbation_score'][str(intensity)][selected_track]['top_compounds'])
                    temp = temp.sort_values(by='pval_adjusted',ascending=True)
                    return temp[:top_n]
        else:
            if selected_track == None:
                if 'top_compounds' not in adata.uns['drug_perturbation_score'][str(intensity)]['overall'].keys():
                    if 'down_compounds' in adata.uns['drug_perturbation_score'][str(intensity)]['overall'].keys():
                        print('All pertubred compounds are not statistically significant!')
                        print('Here are some top insiginifcant compounds, but they are not statistically significant:')
                        return pd.DataFrame.from_dict(adata.uns['drug_perturbation_score'][str(intensity)]['overall']['down_compounds'])[:top_n]
                temp = pd.DataFrame.from_dict(adata.uns['drug_perturbation_score'][str(intensity)]['overall']['top_compounds'])
                temp = temp.sort_values(by='pval_adjusted',ascending=True)
                return temp[:top_n]
            else:
                if selected_track not in adata.uns['drug_perturbation_score'][str(intensity)].keys():
                    raise ValueError('Not a valid track!')
                if 'top_compounds' not in adata.uns['drug_perturbation_score'][str(intensity)][selected_track].keys():
                    if 'down_compounds' in adata.uns['drug_perturbation_score'][str(intensity)][selected_track].keys():
                        print('All pertubred compounds are not statistically significant!')
                        print('Here are some top insiginifcant compounds, but they are not statistically significant:')
                        temp = pd.DataFrame.from_dict(adata.uns['drug_perturbation_score'][str(intensity)][selected_track]['down_compounds'])
                        temp = temp.sort_values(by='pval_adjusted',ascending=True)
                        return temp[:top_n]
                temp = pd.DataFrame.from_dict(adata.uns['drug_perturbation_score'][str(intensity)][selected_track]['top_compounds'])
                temp = temp.sort_values(by='pval_adjusted',ascending=True)
                return temp[:top_n]
    elif cutoff is not None:
        if 'overall' not in adata.uns['drug_perturbation_score'][str(intensity)].keys():
            dict_results = {}
            if selected_track == None:
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
                if selected_track not in adata.uns['drug_perturbation_score'][str(intensity)].keys():
                    raise ValueError('Not a valid track!')
                else:
                    temp = pd.DataFrame.from_dict(adata.uns['drug_perturbation_score'][str(intensity)][selected_track]['top_compounds'])
                    temp = temp.sort_values(by='pval_adjusted',ascending=True)
                    return temp[temp['pval_adjusted'] < cutoff]

        else:
            if selected_track == None:
                if 'top_compounds' not in adata.uns['drug_perturbation_score'][str(intensity)]['overall'].keys():
                    if 'down_compounds' in adata.uns['drug_perturbation_score'][str(intensity)]['overall'].keys():
                        print('All pertubred compounds are not statistically significant!')
                        print('Here are some top insiginifcant compounds that are not statistically significant:')
                        return pd.DataFrame.from_dict(adata.uns['drug_perturbation_score'][str(intensity)]['overall']['down_compounds'])
                temp = pd.DataFrame.from_dict(adata.uns['drug_perturbation_score'][str(intensity)]['overall']['top_compounds'])
                return temp[temp['pval_adjusted'] < cutoff]
            else:
                if selected_track not in adata.uns['drug_perturbation_score'][str(intensity)].keys():
                    raise ValueError('Not a valid track!')
                if 'top_compounds' not in adata.uns['drug_perturbation_score'][str(intensity)][selected_track].keys():
                    if 'down_compounds' in adata.uns['drug_perturbation_score'][str(intensity)][selected_track].keys():
                        print('All pertubred compounds are not statistically significant!')
                        print('Here are some top insiginifcant compounds that are not statistically significant:')
                        return pd.DataFrame.from_dict(adata.uns['drug_perturbation_score'][str(intensity)][selected_track]['down_compounds'])
                temp = pd.DataFrame.from_dict(adata.uns['drug_perturbation_score'][str(intensity)][selected_track]['top_compounds'])
                return temp[temp['pval_adjusted'] < cutoff]
        
    else:
        print('Please specify top_n or cutoff')

def get_top_single_genes(adata, intensity, top_n=None, cutoff=None, selected_track=None):
    '''
    Get top single gene predictions after single gene perturbations at a given intensity.

    Parameters
    ----------
    adata : AnnData object
        Annotated data matrix.
    intensity : int
        Pertubration intensity.
    top_n : int, optional
        Number of top genes to return. The default is None.
    cutoff : float, optional
        P-value cutoff. The default is None.
    selected_track: str, optional
        Show the resultsof track `selected_track`, must run perturbation for the track! The default is None: show overall results.
    '''
    if 'single_gene_perturbation_score' not in adata.uns.keys():
        raise ValueError('Please run single-gene perturbation first!')
    if top_n is not None:
        if 'overall' not in adata.uns['single_gene_perturbation_score'][str(intensity)].keys():
            dict_results = {}
            if selected_track == None:
                for each_track in adata.uns['single_gene_perturbation_score'][str(intensity)].keys():
                    if 'top_compounds' not in adata.uns['single_gene_perturbation_score'][str(intensity)][each_track].keys():
                        print('All pertubred genes are not statistically significant in track %s!'%(each_track))
                        dict_results[each_track] = {}
                    else:
                        dict_results[each_track] = pd.DataFrame.from_dict(adata.uns['single_gene_perturbation_score'][str(intensity)][each_track]['top_compounds'])[:top_n]
                print('You are checking the perturbation results for individual tracks, the returning results are stored in the dictionary.\n You can access the results by using the track name as the key.')
                print('Here are the keys for the dictionary:')
                for key in dict_results.keys():
                    print(key)
                return dict_results
            else:
                if selected_track not in adata.uns['single_gene_perturbation_score'][str(intensity)].keys():
                    raise ValueError('Not a valid track!')
                else:
                    temp = pd.DataFrame.from_dict(adata.uns['single_gene_perturbation_score'][str(intensity)][selected_track]['top_compounds'])
                    temp = temp.sort_values(by='pval_adjusted',ascending=True)
                    temp.rename(columns={'compound': 'gene'}, inplace=True)
                    if 'drug_regulation' in temp.columns:
                        temp = temp.drop(columns=['drug_regulation'])
                    return temp[:top_n]
        else:
            if selected_track == None:
                if 'top_compounds' not in adata.uns['single_gene_perturbation_score'][str(intensity)]['overall'].keys():
                    if 'down_compounds' in adata.uns['single_gene_perturbation_score'][str(intensity)]['overall'].keys():
                        print('All pertubred genes are not statistically significant!')
                        print('Here are some top insiginifcant genes, but they are not statistically significant:')
                        return pd.DataFrame.from_dict(adata.uns['single_gene_perturbation_score'][str(intensity)]['overall']['down_compounds'])[:top_n]
                temp = pd.DataFrame.from_dict(adata.uns['single_gene_perturbation_score'][str(intensity)]['overall']['top_compounds'])
                temp = temp.sort_values(by='pval_adjusted',ascending=True)
                temp.rename(columns={'compound': 'gene'}, inplace=True)
                if 'drug_regulation' in temp.columns:
                    temp = temp.drop(columns=['drug_regulation'])
                return temp[:top_n]
            else:
                if selected_track not in adata.uns['single_gene_perturbation_score'][str(intensity)].keys():
                    raise ValueError('Not a valid track!')
                if 'top_compounds' not in adata.uns['single_gene_perturbation_score'][str(intensity)][selected_track].keys():
                    if 'down_compounds' in adata.uns['single_gene_perturbation_score'][str(intensity)][selected_track].keys():
                        print('All pertubred compounds are not statistically significant!')
                        print('Here are some top insiginifcant compounds, but they are not statistically significant:')
                        temp = pd.DataFrame.from_dict(adata.uns['single_gene_perturbation_score'][str(intensity)][selected_track]['down_compounds'])
                        temp = temp.sort_values(by='pval_adjusted',ascending=True)
                        temp.rename(columns={'compound': 'gene'}, inplace=True)
                        if 'drug_regulation' in temp.columns:
                            temp = temp.drop(columns=['drug_regulation'])
                        return temp[:top_n]
                temp = pd.DataFrame.from_dict(adata.uns['single_gene_perturbation_score'][str(intensity)][selected_track]['top_compounds'])
                temp = temp.sort_values(by='pval_adjusted',ascending=True)
                temp.rename(columns={'compound': 'gene'}, inplace=True)
                if 'drug_regulation' in temp.columns:
                    temp = temp.drop(columns=['drug_regulation'])
                return temp[:top_n]
    elif cutoff is not None:
        if 'overall' not in adata.uns['single_gene_perturbation_score'][str(intensity)].keys():
            dict_results = {}
            if selected_track == None:
                for each_track in adata.uns['single_gene_perturbation_score'][str(intensity)].keys():
                    if 'top_compounds' not in adata.uns['single_gene_perturbation_score'][str(intensity)][each_track].keys():
                        print('All pertubred genes are not statistically significant in track %s!'%(each_track))
                        dict_results[each_track] = {}
                    else:
                        temp = pd.DataFrame.from_dict(adata.uns['single_gene_perturbation_score'][str(intensity)][each_track]['top_compounds'])
                        dict_results[each_track] = temp[temp['pval_adjusted'] < cutoff]
                print('You are checking the perturbation results for individual tracks, the returning results are stored in the dictionary.\n You can access the results by using the track name as the key.')
                print('Here are the keys for the dictionary:')
                for key in dict_results.keys():
                    print(key)
                return dict_results
            else:
                if selected_track not in adata.uns['single_gene_perturbation_score'][str(intensity)].keys():
                    raise ValueError('Not a valid track!')
                else:
                    temp = pd.DataFrame.from_dict(adata.uns['single_gene_perturbation_score'][str(intensity)][selected_track]['top_compounds'])
                    temp = temp.sort_values(by='pval_adjusted',ascending=True)
                    temp.rename(columns={'compound': 'gene'}, inplace=True)
                    if 'drug_regulation' in temp.columns:
                        temp = temp.drop(columns=['drug_regulation'])
                    return temp[temp['pval_adjusted'] < cutoff]

        else:
            if selected_track == None:
                if 'top_compounds' not in adata.uns['single_gene_perturbation_score'][str(intensity)]['overall'].keys():
                    if 'down_compounds' in adata.uns['single_gene_perturbation_score'][str(intensity)]['overall'].keys():
                        print('All pertubred genes are not statistically significant!')
                        print('Here are some top insiginifcant genes that are not statistically significant:')
                        return pd.DataFrame.from_dict(adata.uns['single_gene_perturbation_score'][str(intensity)]['overall']['down_compounds'])
                temp = pd.DataFrame.from_dict(adata.uns['single_gene_perturbation_score'][str(intensity)]['overall']['top_compounds'])
                temp.rename(columns={'compound': 'gene'}, inplace=True)
                if 'drug_regulation' in temp.columns:
                    temp = temp.drop(columns=['drug_regulation'])
                return temp[temp['pval_adjusted'] < cutoff]
            else:
                if selected_track not in adata.uns['single_gene_perturbation_score'][str(intensity)].keys():
                    raise ValueError('Not a valid track!')
                if 'top_compounds' not in adata.uns['single_gene_perturbation_score'][str(intensity)][selected_track].keys():
                    if 'down_compounds' in adata.uns['single_gene_perturbation_score'][str(intensity)][selected_track].keys():
                        print('All pertubred genes are not statistically significant!')
                        print('Here are some top insiginifcant genes that are not statistically significant:')
                        return pd.DataFrame.from_dict(adata.uns['single_gene_perturbation_score'][str(intensity)][selected_track]['down_compounds'])
                temp = pd.DataFrame.from_dict(adata.uns['single_gene_perturbation_score'][str(intensity)][selected_track]['top_compounds'])
                temp.rename(columns={'compound': 'gene'}, inplace=True)
                if 'drug_regulation' in temp.columns:
                    temp = temp.drop(columns=['drug_regulation'])
                return temp[temp['pval_adjusted'] < cutoff]
    else:
        print('Please specify top_n or cutoff')