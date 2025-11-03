import numpy as np
import scanpy as sc
import scipy
import pandas as pd
from scipy.stats import norm
from .attribute_utils import get_data_file_path
def calculateDataPathwayOverlapGene(adata, customized_pathway=None):
    if customized_pathway is None:
        data_path = get_data_file_path('gesa_pathways.npy')
    else:
        data_path = customized_pathway

    pathways = dict(np.load(data_path,allow_pickle=True).tolist())
    adata = adata
    genenames = adata.var.index.tolist()
    out = {}
    for each in list(pathways.keys()):
        temp = []
        for gene in pathways[each]:
            if gene in genenames and gene not in temp:
                temp.append(gene)
        if len(temp) >0:
            out[each] = temp
    tmp = {}
    for key, value in out.items():
        value = '!'.join(value)
        if value in tmp:
            tmp[value].append(key)
        else:
            tmp[value] = [ key ]
    out = {}
    for value, keys in tmp.items():
        out[','.join(keys)] = value.split('!')
    # out_path = os.path.join(source_folder,str(ITERATION),'data_pathway_overlap_genes.npy')
    # np.save(out_path,out)
    specific_gene_len_dict = {}
    data_pathway_overlap_genes = out
    for each in list(data_pathway_overlap_genes.keys()):
        specific_gene_len_dict[each] = len(data_pathway_overlap_genes[each])
    adata.uns['data_pathway_overlap_genes'] = out
    adata.uns['pathway_gene_len_dict'] = specific_gene_len_dict
    # adata.write('./iterativeTrainingNOV26/'+str(ITERATION)+'/stagedata/dataset.h5ad')
    # out_path = os.path.join(source_folder,str(ITERATION),'pathway_gene_len_dict.npy')
    # np.save(out_path,specific_gene_len_dict)
    return adata
def calculateTopPathwayGeneRanking(adata):
    '''
    rank pathways based on gene weights
    '''
    pathway_gene = adata.uns['data_pathway_overlap_genes']

    avg_ranking = {}
    for i in range(1,4):
        # adata = sc.read_h5ad(f'./iterativeTrainingNOV26/'+str(ITERATION)+'/stagedata/%d.h5ad'%i)
        stageadataids = adata.obs[adata.obs['stage']==i].index.tolist()
        stageadata = adata[stageadataids]
        avg_ranking[i] = {}
        adata_genes = stageadata.var.index.values.tolist()
        adata_genes_index_dict={}
        for idx,gene in enumerate(adata_genes):
            adata_genes_index_dict[gene] = idx
        for clusterid in set(stageadata.obs['leiden']):
            
            avg_ranking[i][clusterid] = {}
            clusteradataid = stageadata.obs[stageadata.obs['leiden']==clusterid].index.tolist()
            clusteradata = stageadata[clusteradataid]
            cluster_gene_weight_table = clusteradata.layers['geneWeight']
            avg_cluster_gene_weight_table = np.mean(cluster_gene_weight_table,axis=0).reshape(-1)
    
            
            avg_geneWeightTable_ranking = scipy.stats.rankdata(avg_cluster_gene_weight_table)
            avg_geneWeightTable_ranking = np.zeros_like(avg_geneWeightTable_ranking)+len(adata_genes)-avg_geneWeightTable_ranking
    
            for pathway in list(pathway_gene.keys()):
                avg_ranking[i][clusterid][pathway] = 0
                for gene_in_pathway in pathway_gene[pathway]:
                    avg_ranking[i][clusterid][pathway]+=avg_geneWeightTable_ranking[adata_genes_index_dict[gene_in_pathway]]
                avg_ranking[i][clusterid][pathway] = avg_ranking[i][clusterid][pathway]/len(pathway_gene[pathway])
    new_av_ranking = {}
    for i in list(avg_ranking.keys()):
        new_av_ranking[str(i)] = {}
        for j in list(avg_ranking[i].keys()):
            
            new_av_ranking[str(i)][str(j)]={k: idx+1 for idx,(k, v) in enumerate(sorted(avg_ranking[i][j].items(), key=lambda item: item[1]))}
    # np.save('./iterativeTrainingNOV26/'+str(ITERATION)+'/avg_ranking.npy',new_av_ranking)
    adata.uns['pathway_ranking'] = new_av_ranking
    return adata
def findTopGenesInCluster(adata,pval=0.1):
    '''
    find top genes with high gene weight in each cluster (pval < 0.1 default)
    '''
    topClusterGeneBasedOnGeneWeight = {}
    for i in range(1,4):
        stageadataids = adata.obs[adata.obs['stage']==i].index.tolist()
        stageadata = adata[stageadataids]
        topClusterGeneBasedOnGeneWeight[i] = {}
        adata_genes = adata.var.index.values.tolist()
        adata_genes_index_dict={}
        for idx,gene in enumerate(adata_genes):
            adata_genes_index_dict[gene] = idx
        for clusterid in set(stageadata.obs['leiden']):
            topClusterGeneBasedOnGeneWeight[i][clusterid] = []
            clusteradataid = stageadata.obs[stageadata.obs['leiden']==clusterid].index.tolist()
            clusteradata = stageadata[clusteradataid]
            cluster_gene_weight_table = clusteradata.layers['geneWeight']
            avg_cluster_gene_weight_table = np.mean(cluster_gene_weight_table,axis=0).reshape(-1)
            dist_mean = np.mean(avg_cluster_gene_weight_table)
            dist_std = np.std(avg_cluster_gene_weight_table)
            for idx, eachgene in enumerate(avg_cluster_gene_weight_table):
                cdf = 1 - norm.cdf(eachgene, dist_mean, dist_std)
                if cdf < pval:
                    topClusterGeneBasedOnGeneWeight[i][clusterid].append(adata_genes[idx])
    adata.uns['topClusterGeneBasedOnGeneWeight'] = topClusterGeneBasedOnGeneWeight
    return adata
def merge_drugs_with_sametarget_samedirection(adata,overlapped_drug_direction_profile):
    reverse = {}
    for key in list(overlapped_drug_direction_profile.keys()):
        temp = ''
        for value in overlapped_drug_direction_profile[key]:
            temp+=str(value)+','
        temp = temp[:-1]
        if temp not in reverse:
            reverse[temp] = ''+str(key)
        else:
            reverse[temp]+=','+str(key)
    for key in list(reverse.keys()):
        if key == '':
            del reverse[key]
    out = {}
    drug_len = {}
    for key in list(reverse.keys()):
        out[reverse[key]] = []
        for each in key.split(','):
            out[reverse[key]].append(each)
        drug_len[reverse[key]] = len(reverse[key].split(','))
   
    adata.uns['data_drug_overlap_genes'] = out
    adata.uns['drug-gene_len_dict'] = drug_len
    return adata

import gc
from scipy.sparse import csr_matrix
def alterative(source_directory):
    for i in range(4):
        temp1 = sc.read_h5ad(source_directory / 'stagedata' / '%d.h5ad'%i)
        temp2 = sc.read_h5ad(source_directory / 'stagedata' / 'concat_%d.h5ad'%i)
        temp1.obsp['gcn_connectivities'] = temp2.obsp['gcn_connectivities'].copy()
        temp1.layers['concat'] = csr_matrix(temp2.layers['concat'].copy())
        
        temp2 = None
        gc.collect()
        temp1.write(source_directory / 'stagedata' / '%d.h5ad'%i, compression='gzip', compression_opts=9)
        

def calculateDrugOverlapGene(adata, cmap_df=None, drug_profile_directory=None):
    genenames = adata.var.index.tolist()
    # use customized drugs
    if drug_profile_directory is not None:
        drug = dict(np.load(drug_profile_directory,
                    allow_pickle=True).tolist())
        out = {}
        for each in list(drug.keys()):
            temp = []
            temp_genes = []
            # for gene in drug[each]:
                # gene = gene.split(':')[0]
                # if gene in genenames and gene not in temp:
                #     temp.append(gene)
            temp_genes = [s.partition(':')[0] for s in drug[each]]
                
            temp = list(set(temp_genes) & set(genenames))

            if len(temp) > 0:
                out[each] = temp
    
        tmp = {}
        for key, value in out.items():
            value = '!'.join(value)
            if value in tmp:
                tmp[value].append(key)
            else:
                tmp[value] = [key]
        out = {}
        for value, keys in tmp.items():
            out[','.join(keys)] = value.split('!')
        specific_gene_len_dict = {}
        data_pathway_overlap_genes = out
        for each in list(data_pathway_overlap_genes.keys()):
            specific_gene_len_dict[each] = len(
                data_pathway_overlap_genes[each])
  
        adata.uns['data_drug_overlap_genes'] = out
        adata.uns['drug-gene_len_dict'] = specific_gene_len_dict
        return adata
    elif cmap_df is not None:
        cmap_df = cmap_df[cmap_df.columns.intersection(genenames)]
        cols = cmap_df.columns.tolist()
        rows = cmap_df.index.tolist()

        data_pathway_overlap_genes = {row: cols for row in rows}
        specific_gene_len_dict = {compound: len(
            data_pathway_overlap_genes[compound]) for compound in data_pathway_overlap_genes}

        adata.uns['data_drug_overlap_genes'] = data_pathway_overlap_genes
        adata.uns['drug-gene_len_dict'] = specific_gene_len_dict
        return adata


def merge_drugs_with_sametarget_samedirection(adata, overlapped_drug_direction_profile=None):
    reverse = {}
    for key in list(overlapped_drug_direction_profile.keys()):
        temp = ''
        for value in overlapped_drug_direction_profile[key]:
            temp += str(value)+','
        temp = temp[:-1]
        if temp not in reverse:
            reverse[temp] = ''+str(key)
        else:
            reverse[temp] += ','+str(key)
    for key in list(reverse.keys()):
        if key == '':
            del reverse[key]
    out = {}
    drug_len = {}
    for key in list(reverse.keys()):
        out[reverse[key]] = []
        for each in key.split(','):
            out[reverse[key]].append(each)
        drug_len[reverse[key]] = len(reverse[key].split(','))

    adata.uns['data_drug_overlap_genes'] = out
    adata.uns['drug-gene_len_dict'] = drug_len
    return adata


def assign_drug_direction(adata, cmap_df=None, customized_drug_direction=None):
    # if cmap_df is not None:
    data_path = get_data_file_path('brdID2cmapName.npy')
    brdID2cmapName = dict(np.load(data_path, allow_pickle=True).tolist())
    target_file = adata.uns['data_drug_overlap_genes']
    # assign direciton of drugs to regulate genes based on CMAP direction profile
    if customized_drug_direction is not None:
        CMAP_direction_profile = dict(
            np.load(customized_drug_direction, allow_pickle=True).tolist())
        out = {}
        for each in list(target_file.keys()):  # each='compound1;compound2;'
            lst = each.split(',')  # lst = ['compound1','compound2']
            for item in lst:  # item = 'compound1'
                if item in list(CMAP_direction_profile.keys()):
                    selected_drug = item
                elif brdID2cmapName[item] in list(CMAP_direction_profile.keys()):
                    selected_drug = brdID2cmapName[item]
                else:
                    continue
                temp = CMAP_direction_profile[selected_drug]
                if item not in out:
                    out[item] = []
                for gene in temp:
                    if gene.split(':')[0] in target_file[each]:
                        out[item].append(gene)
        adata = merge_drugs_with_sametarget_samedirection(adata, out)
        return adata
    else:
        out = {}
        
        for each in list(target_file.keys()):  # each='compound1;compound2;'
            lst = each.split(',')  # lst = ['compound1','compound2']
            for item in lst:  # item = 'compound1'
                if item in cmap_df.index.tolist():
                    selected_drug = item
                    break
                elif brdID2cmapName[item] in cmap_df.index.tolist():
                    selected_drug = brdID2cmapName[item]
                    break
                else:
                    continue
            temp = cmap_df.loc[selected_drug]
            temp_record = []
            
            for gene in target_file[each]:
                if gene in cmap_df.columns.tolist():
                    temp_record.append(str(gene)+':'+str(temp[gene]))
            
            out[each] = temp_record
        adata = merge_drugs_with_sametarget_samedirection(adata, out)
        return adata

def process_customized_drug_database(data, customized_drug):
    drug = dict(np.load(customized_drug, allow_pickle=True).tolist())
    gene_names = data.var.index.tolist()
   # upper case gene names
    gene_names = [gene.upper() for gene in gene_names]
    out = {}
    for each in list(drug.keys()):
        temp = []
        for gene in drug[each]:
            gene1 = gene.split(':')[0]
            #upper case
            gene1 = gene1.upper()
            if gene1 in gene_names and gene not in temp:
                temp.append(gene)
        if len(temp) > 0:
            out[each] = temp

    tmp = {}
    for key, value in out.items():
        value = '!'.join(value)
        if value in tmp:
            tmp[value].append(key)
        else:
            tmp[value] = [key]
    out = {}
    for value, keys in tmp.items():
        out[','.join(keys)] = value.split('!')
    specific_gene_len_dict = {}
    data_pathway_overlap_genes = out
    for each in list(data_pathway_overlap_genes.keys()):
        specific_gene_len_dict[each] = len(data_pathway_overlap_genes[each])

    data.uns['data_drug_overlap_genes'] = out
    data.uns['drug-gene_len_dict'] = specific_gene_len_dict
    return data
                                     
def find_overlap_and_assign_direction(adata, customized_direction=None, customized_drug=None,cmap_dir=None):
    load_cmap = False
    if customized_drug is None or customized_direction is None:
        if cmap_dir is None:
            raise ValueError("cmap_dir must be provided! Please download the CMAPDirectionalDf.npy provided in the GitHub repo.")
        arr = np.load(cmap_dir, allow_pickle=True)
        cmap_df = pd.DataFrame(arr[0], index=arr[1], columns=arr[2])
        load_cmap = True


    if customized_drug is None and customized_direction is None:
        print("using cmap drug profile & direction")
        genenames = adata.var.index.tolist()

        cmap_df = cmap_df[cmap_df.columns.intersection(genenames)]
        cols = cmap_df.columns

        out = {row: cols+':'+cmap_df.loc[row]
               for row in cmap_df.index.tolist()}
        print('len(cmap_df.columns)', len(cmap_df.columns))
        nGenes = len(cmap_df.columns)
        drug_len = {row: nGenes for row in cmap_df.index.tolist()}
        adata.uns['drug-gene_len_dict'] = drug_len

        adata = merge_drugs_with_sametarget_samedirection(adata, out)
    else:
        if (customized_drug != None):
            
            print("using customized drug profile:", customized_drug)
            adata = calculateDrugOverlapGene(
                adata, drug_profile_directory=customized_drug)
        else:
            print("using cmap drug profile")
            if cmap_dir is None:
                raise ValueError("cmap_dir must be provided! Please download the CMAPDirectionalDf.npy provided in the GitHub repo.")
            if load_cmap == False:
                arr = np.load(cmap_dir, allow_pickle=True)
                cmap_df = pd.DataFrame(arr[0], index=arr[1], columns=arr[2])
                load_cmap = True
            genenames = adata.var.index.tolist()
            cmap_df = cmap_df[cmap_df.columns.intersection(genenames)]
            adata = calculateDrugOverlapGene(adata, cmap_df=cmap_df)

        if (customized_direction != None):
            print("using cutomized drug dirction:", customized_direction)
            adata = assign_drug_direction(
                adata, customized_drug_direction=customized_direction)
        else:
            print("using cmap drug direction")
            adata = assign_drug_direction(adata, cmap_df=cmap_df)

    return adata
import numpy as np
import scanpy as sc
import pickle as pkl
class TreeNode:
    def __init__(self, name):
        self.name = name
        self.children = []

    def add_child(self, child):
        self.children.append(child)

def print_tree(node, level=0):
    print(" " * (4 * level) + "Stage %d|-- "%(level) + node.name)
    for child in node.children:
        # print(child.name)
        print_tree(child, level + 1)

def getClusterPaths_with_cell_types(edges, total_stages,cell_types):
    '''
    Obtain the paths of each cluster for multiple stages with cell type information.
    
    parameters
    -----------
    edges: list
        A list of lists, where each sublist contains edges between consecutive stages.
    total_stages: int
        Total number of stages.
    cell_types: dict
        A dictionary containing cell types for each stage.

    return
    -----------
    paths: list
        A collection of paths of clusters.
    '''
    if len(edges) != total_stages - 1:
        raise ValueError("Number of edges must be one less than total stages")

    paths = {}
    for key in list(edges.keys()):
        edges[int(key)] = edges[key]
    # Initialize paths with the first set of edges
    for each in edges[0]:
        if str(each[0]) not in paths:
            root_name = cell_types[str(0)][int(each[0])]
            leaf_name = cell_types[str(1)][int(each[1])]
            root = TreeNode(str(each[0])+'_'+root_name)
            tree_node = TreeNode(str(each[1])+'_'+leaf_name)
            root.add_child(tree_node)
            paths[str(each[0])] = [[root], {str(each[1]):tree_node}]
        else:
            leaf_name = cell_types[str(1)][int(each[1])]
            tree_node = TreeNode(str(each[1])+'_'+leaf_name)
            paths[str(each[0])][0][0].add_child(tree_node)
            paths[str(each[0])][1][str(each[1])] = tree_node
    # Iterate through remaining stages
    for stage in range(1, total_stages - 1):
        for each in edges[stage]:
            for item in paths.keys():
                if len(paths[item]) == stage:
                    continue
                try:
                    if str(each[0]) in list(paths[item][stage].keys()):
                        leaf_name = cell_types[str(stage+1)][int(each[1])]
                        tree_node = TreeNode(str(each[1])+'_'+leaf_name)
                        paths[item][stage][str(each[0])].add_child(tree_node)
                        if len(paths[item]) == stage + 1:
                            paths[item].append({})
                            paths[item][stage + 1][str(each[1])] = tree_node
                        else:
                            paths[item][stage + 1][str(each[1])] = tree_node
                except:
                    pass
    return paths

def visualize_dynamic_graphs_by_text(edges,cell_types, write=False):
    '''
    Visualize the dynamic graphs by text.

    parameters
    -----------
    edges: list
        A list of lists, where each sublist contains edges between consecutive stages.
    cell_types: dict
        A dictionary containing cell types for each stage.
    write: bool
        Whether to save the results to a file.

    return
    -----------
    None
    '''
    
    edges_keys = list(edges.keys())
    for k in edges_keys:
        edges[int(k)] = edges[k]
    # convert keys to int and sort
    edges_keys = [int(k) for k in edges_keys]
    # remove duplicates and sort
    edges_keys = sorted(list(set(edges_keys)))
    edges = {int(k): edges[k] for k in edges_keys}
    paths = getClusterPaths_with_cell_types(edges,len(edges_keys)+1,cell_types)
    if not write:
        for each in list(paths.keys()):
            print('Track '+each+' :')
            print_tree(paths[each][0][0])
    if write:
        from contextlib import redirect_stdout

        with open("temp_dynamic_graphs.txt", "a+", encoding="utf-8") as f:
            for each in list(paths.keys()):
                f.write('Track '+each+' :\n')
                with redirect_stdout(f):
                    print_tree(paths[each][0][0]) 
        print('Please check the temp_dynamic_graphs.txt!') 
