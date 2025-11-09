import pandas as pd
def get_top_pathways(adata, perturb_change, top_n=None, cutoff=None,selected_track=None):
    """
    Get top pathways predictions after pathway perturbations at a given perturb_change.

    Parameters
    ----------
    adata : AnnData object
        Annotated data matrix.
    perturb_change : int
        perturbation perturb_change.
    top_n : int, optional
        Number of top pathways to return. The default is None.
    cutoff : float, optional
        P-value cutoff. The default is None.
    selected_track: str, optional
        Show the resultsof track `selected_track`, must run perturbation for the track! The default is None: show overall results.
    """


    if top_n is not None:
        if 'overall' not in adata.uns['pathway_perturbation_score'][str(perturb_change)].keys():
            dict_results = {}
            if selected_track == None:
                for each_track in adata.uns['pathway_perturbation_score'][str(perturb_change)].keys():
                    if 'top_compounds' not in adata.uns['pathway_perturbation_score'][str(perturb_change)][each_track].keys():
                        print('All pertubred pathways are not statistically significant in track %s!'%(each_track))
                        dict_results[each_track] = {}
                    else:
                        temp = pd.DataFrame.from_dict(adata.uns['pathway_perturbation_score'][str(perturb_change)][each_track]['top_compounds'])[:top_n]
                        temp.rename(columns={'compound': 'pathways'}, inplace=True)
                        dict_results[each_track] = temp
                print('You are checking the perturbation results for individual tracks, the returning results are stored in the dictionary.\n You can access the results by using the track name as the key.')
                print('Here are the keys for the dictionary:')
                for key in dict_results.keys():
                    print(key)
                return dict_results
            else:
                if selected_track not in adata.uns['pathway_perturbation_score'][str(perturb_change)].keys():
                    raise ValueError('Not a valid track! Valid tracks are: %s' % (list(adata.uns['pathway_perturbation_score'][str(perturb_change)].keys())))
                else:
                    temp = pd.DataFrame.from_dict(adata.uns['pathway_perturbation_score'][str(perturb_change)][selected_track]['top_compounds'])
                    temp.rename(columns={'compound': 'pathways'}, inplace=True)
                    temp = temp.sort_values(by='pval_adjusted',ascending=True)
                    return temp[:top_n]
        
        else:
            if selected_track == None:
                if 'top_compounds' not in adata.uns['pathway_perturbation_score'][str(perturb_change)]['overall'].keys():
                    print('All pertubred pathways are not statistically significant!')
                    print('Here are the top %s pathways that are not statistically significant:'%(str(top_n)))
                    temp = pd.DataFrame.from_dict(adata.uns['pathway_perturbation_score'][str(perturb_change)]['overall']['down_compounds'])

                
                else:
                    temp = pd.DataFrame.from_dict(adata.uns['pathway_perturbation_score'][str(perturb_change)]['overall']['top_compounds'])
                temp = temp.sort_values(by='pval_adjusted',ascending=True)
                temp.rename(columns={'compound': 'pathways'}, inplace=True)
                temp.rename(columns={'drug_regulation': 'regulated genes'}, inplace=True)
                return temp[:top_n]
            else:
                if selected_track not in adata.uns['pathway_perturbation_score'][str(perturb_change)].keys():
                    raise ValueError('Not a valid track! Valid tracks are: %s' % (list(adata.uns['pathway_perturbation_score'][str(perturb_change)].keys())))
                if 'top_compounds' not in adata.uns['pathway_perturbation_score'][str(perturb_change)][selected_track].keys():
                    print('All pertubred pathways are not statistically significant!')
                    print('Here are the top %s pathways that are not statistically significant:'%(str(top_n)))
                    temp = pd.DataFrame.from_dict(adata.uns['pathway_perturbation_score'][str(perturb_change)][selected_track]['down_compounds'])
                
                else:
                    temp = pd.DataFrame.from_dict(adata.uns['pathway_perturbation_score'][str(perturb_change)][selected_track]['top_compounds'])
                temp = temp.sort_values(by='pval_adjusted',ascending=True)
                temp.rename(columns={'compound': 'pathways'}, inplace=True)
                temp.rename(columns={'drug_regulation': 'regulated genes'}, inplace=True)
                return temp[:top_n]
    elif cutoff is not None:
        if 'overall' not in adata.uns['pathway_perturbation_score'][str(perturb_change)].keys():
            dict_results = {}
            if selected_track == None:
                for each_track in adata.uns['pathway_perturbation_score'][str(perturb_change)].keys():
                    if 'top_compounds' not in adata.uns['pathway_perturbation_score'][str(perturb_change)][each_track].keys():
                        print('All pertubred compounds are not statistically significant in track %s!'%(each_track))
                        dict_results[each_track] = {}
                    else:
                        temp = pd.DataFrame.from_dict(adata.uns['pathway_perturbation_score'][str(perturb_change)][each_track]['top_compounds'])
                        temp = temp.sort_values(by='pval_adjusted',ascending=True)
                        temp.rename(columns={'compound': 'pathways'}, inplace=True)
                        dict_results[each_track] = temp[temp['pval_adjusted'] < cutoff]
                print('You are checking the perturbation results for individual tracks, the returning results are stored in the dictionary.\n You can access the results by using the track name as the key.')
                print('Here are the keys for the dictionary:')
                for key in dict_results.keys():
                    print(key)
                return dict_results
            else:
                if selected_track not in adata.uns['pathway_perturbation_score'][str(perturb_change)].keys():
                    raise ValueError('Not a valid track! Valid tracks are: %s' % (list(adata.uns['pathway_perturbation_score'][str(perturb_change)].keys())))
                else:
                    temp = pd.DataFrame.from_dict(adata.uns['pathway_perturbation_score'][str(perturb_change)][selected_track]['top_compounds'])
                    temp.rename(columns={'compound': 'pathways'}, inplace=True)
                    temp = temp.sort_values(by='pval_adjusted',ascending=True)
                    return temp[temp['pval_adjusted'] < cutoff]
        
        else:
            if selected_track == None:
                if 'top_compounds' not in adata.uns['pathway_perturbation_score'][str(perturb_change)]['overall'].keys():
                    print('All pertubred pathways are not statistically significant!')
                    print('Here are the pathways that are not statistically significant:')
                    temp = pd.DataFrame.from_dict(adata.uns['pathway_perturbation_score'][str(perturb_change)]['overall']['down_compounds'])

                else:
                    temp = pd.DataFrame.from_dict(adata.uns['pathway_perturbation_score'][str(perturb_change)]['overall']['top_compounds'])
                temp = temp.sort_values(by='pval_adjusted',ascending=True)
                temp.rename(columns={'compound': 'pathways'}, inplace=True)
                temp.rename(columns={'drug_regulation': 'regulated genes'}, inplace=True)
                return temp[temp['pval_adjusted'] < cutoff]
            else:
                if selected_track not in adata.uns['pathway_perturbation_score'][str(perturb_change)].keys():
                    raise ValueError('Not a valid track! Valid tracks are: %s' % (list(adata.uns['pathway_perturbation_score'][str(perturb_change)].keys())))
                if 'top_compounds' not in adata.uns['pathway_perturbation_score'][str(perturb_change)][selected_track].keys():
                    print('All pertubred pathways are not statistically significant!')
                    print('Here are the pathways that are not statistically significant:')
                    temp = pd.DataFrame.from_dict(adata.uns['pathway_perturbation_score'][str(perturb_change)][selected_track]['down_compounds'])
                
                else:
                    temp = pd.DataFrame.from_dict(adata.uns['pathway_perturbation_score'][str(perturb_change)][selected_track]['top_compounds'])
                temp = temp.sort_values(by='pval_adjusted',ascending=True)
                temp.rename(columns={'compound': 'pathways'}, inplace=True)
                temp.rename(columns={'drug_regulation': 'regulated genes'}, inplace=True)
                return temp[temp['pval_adjusted'] < cutoff]
        
    else:
        print('Please specify top_n or cutoff')