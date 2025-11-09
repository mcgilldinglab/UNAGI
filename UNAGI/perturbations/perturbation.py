#this is push back, last cell is push forward
from scipy.sparse import issparse, csr_matrix
import numpy as np
import os
import numba as nb
import random
from pathlib import Path
import gc
from tqdm import tqdm
import json
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
import time
from scipy.spatial import distance
import torch
from scipy.sparse import issparse
#import DataLoader from torch
from torch.utils.data import DataLoader
from ..utils.gcn_utils import setup_graph
import threading
from ..model.models import VAE, Plain_VAE
from .analysis_perturbation import perturbationAnalysis
def get_random_genes(adata, n,seed):
    n = int(n)
    np.random.seed(seed)
    genes = adata.var.index.tolist()
    return np.random.choice(genes,n)
def get_overlap_genes_pathway(adata, pathway):
    genes = adata.var.index
    overlap_genes = []
    for each in pathway:
        each = each.upper()
        if each in genes:
            overlap_genes.append(each)
    return overlap_genes
@nb.njit(cache=True)
def scale_csr_rows(data, indices, indptr, rows, factors):
    """
    Scale the rows `rows` of a CSR matrix in place.
    factors is a dense 1‑D array of length n_cols.
    Only multiplications where factors[col] != 1 are actually done.
    """
    for r in rows:
        row_start = indptr[r]
        row_end   = indptr[r + 1]
        for p in range(row_start, row_end):
            col = indices[p]
            f   = factors[col]
            if f != 1.0:                 # skip genes with factor 1
                data[p] *= f
class perturbator:
    def __init__(self, model_path,data,config_path):
        # torch.load(model_path)
        self.data = data
        if issparse(self.data.X):
            if self.data.X.format != 'csr':
                self.data.X = self.data.X.tocsr(copy=False)
        else:
            self.data.X = csr_matrix(self.data.X, copy=False)
        self.adj = self.data.obsp['gcn_connectivities']
        self.adj = self.adj.asformat('coo')
        self.adj = setup_graph(self.adj)
        self.config = json.load(open(config_path))
        self.adj = self.adj.to(self.config['device'])

        self.data.var_names_make_unique()
        
        self.model = VAE(self.config['input_dim'], self.config['hidden_dim'], self.config['graph_dim'], self.config['latent_dim'],beta=self.config['beta'],distribution=self.config['dist'])
        self.model.load_state_dict(torch.load(model_path,map_location=torch.device(self.config['device'])))
        self.model.to(self.config['device'])
        self.model.eval()
        self.data.obsm['embdedding'] = self._obtain_embedding()
        
        
        
    def _obtain_embedding(self, data = None):
        
        if data is None:
            data = self.data.X
        if issparse(data):
            data = data.toarray()
        cell_loader = DataLoader(data.astype('float32'), batch_size=len(data), shuffle=False, num_workers=0)

        for perturbed_index, x in enumerate(cell_loader):
            x = x.to(self.config['device'])
            z = self.model.get_latent_representation(x,self.adj)
        # z to np array
        if self.config['device'] != 'cpu':
            z = z.detach().cpu().numpy()
        else:
            z = z.detach().numpy()
        return z
    
    def _run_gene_pca(self,adata):
        genes = adata.var_names.tolist()
        adata = adata.X.T
        from sklearn.decomposition import PCA
        pca = PCA(n_components=25)
        pca.fit(adata)
        temp = pca.transform(adata)
        
        pc_genes_df = pd.DataFrame(temp, 
                            index=genes,
                            columns=[f'PC{i+1}' for i in range(temp.shape[1])])
        corr_matrix_genes = pc_genes_df.T.corr()

        return pc_genes_df.T, corr_matrix_genes
    import numpy as np
    def getPPINetworkDict(self, adata, hippiefile = 'hippie_current.txt',stringfile = '9606.protein.links.v11.5.txt',stringinfo = '9606.protein.info.v11.5.txt', output_GIN_path = 'PPINetworkDict.npy'):
        hippiefile = open(hippiefile)
        hippiefile = hippiefile.readlines()
        stringfile = open(stringfile)
        stringfile = stringfile.readlines()
        stringinfo = open(stringinfo)
        stringinfo = stringinfo.readlines()
        
        adatagene = adata.var.index.values.tolist()
        stringinfo_dict = {}
        for each in stringinfo[1:]:
            each = each.split('\t')
            if each[1] in adatagene:
                stringinfo_dict[each[0]] = each[1]
        newhippie = []
        for each in hippiefile[1:]:
            each = each.split('\t')
            A = each[0].split('_')[0]
            B = each[2].split('_')[0]
            if A in adatagene and B in adatagene and float(each[4])>=0.8:
                newhippie.append([A, B, int(float(each[4])*1000)])
                newhippie.append([B, A, int(float(each[4])*1000)])
        string_dict = {}
        human_encode_dict = {}
        newstring = []
        for each in stringfile[1:]:
            each = each.split(' ')
            score = int(each[2].strip('\n'))
            if score>=800:
                if each[0] in stringinfo_dict.keys() and each[1] in stringinfo_dict.keys():
                    gene_source = stringinfo_dict[each[0]]
                    gene_target=stringinfo_dict[each[1]]
                    newstring.append([gene_source,gene_target,score])
        bidirectionalmerge = newstring
        merged_dict = {}
        for each in bidirectionalmerge:
            if each[0] not in merged_dict.keys():
                merged_dict[each[0]] = {}
            merged_dict[each[0]][each[1]] = each[2]/1000
        np.save(output_GIN_path, merged_dict)
    
    def _BFS(self,adata, PPIDict,gene,PCA,coef,genetable):
        table={}
        depth=0
        tableset = set()
        PPIFactor=np.zeros(adata.X.shape[1])
        queue = []
        queue.append({gene:1})
        epicenterid = genetable[gene]
        PPIFactor[epicenterid] = 1
        table[gene]={'depth':0,'distance':0,'parent':None}
        while len(queue) !=0:
            
            go = queue.pop(0)
            popedgenename = list(go.keys())[0]
            
            if popedgenename not in PPIDict.keys():
                continue
            if popedgenename in tableset:
                continue
            tableset.add(popedgenename)
            
            depth=table[popedgenename]['depth']
            if depth > 5:
                break
            #table[go] = depth
            idcurrent=genetable[popedgenename]
            #vecA = adata.X[:, idcurrent].toarray().reshape(-1)
            pcacurrent=PCA[popedgenename]
            for each in PPIDict[popedgenename].keys():
                if each in tableset:
                    continue
                if each not in PPIDict.keys():
                    continue
                if PPIDict[popedgenename][each] < 0.8:
                    continue
                iddecedent = genetable[each]
                idneighbour=genetable[each]
                pcadecendent=PCA[each]
                l2distance = np.linalg.norm(pcacurrent-pcadecendent)
                table[each] = {}
                distance = table[popedgenename]['distance']
                distance+=(1-coef[popedgenename][each])/PPIDict[popedgenename][each]
                # distance+=l2distance/PPIDict[popedgenename][each]
                table[each]['distance']= distance
                # PPIFactor[iddecedent] = np.exp(-0.05*distance)
                PPIFactor[iddecedent] = np.exp(-0.2*distance)
                #vecB=adata.X[:, iddecedent].toarray().reshape(-1)
                correlation = coef[popedgenename][each]#np.corrcoef(pcacurrent,pcadecendent)
                table[each]['parent']=popedgenename
                if correlation<0:
                    PPIFactor[iddecedent] = -PPIFactor[iddecedent]
                table[each]['depth'] = depth+1
                table[each]['score'] = PPIDict[popedgenename][each]
                queue.append({each:PPIDict[popedgenename][each]})

        return PPIFactor
    def preprocess_GIN(self, selected_adata_index, perturb_resource, GIN_path = 'PPINetworkDict.npy'):
        PPIdict = np.load(GIN_path,allow_pickle=True).item()
        adata = self.data
        genetable = {gene:i for i,gene in enumerate(adata.var.index)}
        pca_df, coeff = self._run_gene_pca(adata[selected_adata_index])
        GINFactor = None
        for each_gene in perturb_resource:
            if GINFactor is None:
                GINFactor = [self._BFS(adata[selected_adata_index],PPIdict,each_gene,pca_df,coeff,genetable)]
            else:
                GINFactor.append(self._BFS(adata[selected_adata_index],PPIdict,each_gene,pca_df,coeff,genetable))
        # GINFactor = GINFactor/len(perturb_resource)
        return GINFactor
    
    def change_GIN_targets(self, adata,selected_adata_index, perturb_resource,GIN_path='PPINetworkDict.npy'):
        GINFactors = self.preprocess_GIN(selected_adata_index, perturb_resource,GIN_path)
        temp = adata[selected_adata_index,:].X.copy()
        new_exps = None
        for i, each_gene in enumerate(perturb_resource):
            scale = perturb_resource[each_gene]
            if new_exps is None:
                new_exps = temp + GINFactors[i] * scale#temp.multiply(GINFactors[i] * (scale - 1) + 1) #expression * [GI_factor * (pertubr_scale - 1) + 1] == (expression * pertubr_scale - expression) * GI_factor + expression
            else:
                new_exps += temp + GINFactors[i] * scale#temp.multiply(GINFactors[i] * (scale - 1) + 1) #expression * [GI_factor * (pertubr_scale - 1) + 1] == (expression * pertubr_scale - expression) * GI_factor + expression
        new_exps[new_exps<0] = 0
        adata[selected_adata_index,:].X = new_exps/len(perturb_resource)
        return adata
    def _calculateScore(self,delta,flag,weight=100):
        '''
        Calculate the perturbation score.

        parameters
        -----------
        delta: float
            The perturbation distance.(D(Perturbed cluster, others stages)  - D(Original cluster, others stages)  (in z space))
        flag: int
            The stage of the time-series single-cell data.
        weight: float
            The weight to control the perturbation score.

        return
        --------
        out: float
            The perturbation score.
        '''
        out = 0
        out1 = 0
        separate = []
        for i, each in enumerate(delta):
            
            if i != flag:
                out+=(1-1/(1+np.exp(weight*each*np.sign(i-flag)))-0.5)/0.5
                out1+=np.abs((1-1/(1+np.exp(weight*each))-0.5)/0.5)
        return out/(len(delta)-1), out1/(len(delta)-1)
    def calculate_deltaD_for_tracks(self, list_of_indices):
        """
        list_of_indices     : list[ list[str] ]
                            Each inner list contains obs_names (cell IDs) of a track.
        Returns
        -------
        deltaD : (n_tracks × n_tracks) ndarray  (use .tolist() if you need a Python list)
        """
        adata   = self.data                             # shortcut
        n_tracks = len(list_of_indices)
        dim      = adata.obsm['perturbed_embedding'].shape[1]

        # ---------- 1. Map obs_names → integer row indices (vectorised) ----------
        # adata.obs_names is a pandas Index; get_indexer is C‑level fast
        idx_arrays = [
            adata.obs_names.get_indexer(track_indices)
            for track_indices in list_of_indices
        ]

        # ---------- 2. Compute one mean vector per track (no Python loops in mean) ----------
        P_means = np.empty((n_tracks, dim), dtype=adata.obsm['perturbed_embedding'].dtype)
        O_means = np.empty_like(P_means)

        P_all = adata.obsm['perturbed_embedding']   # cell × dim
        O_all = adata.obsm['embdedding']            # keep author’s original key

        for k, idx in enumerate(idx_arrays):
            P_means[k] = P_all[idx].mean(axis=0)
            O_means[k] = O_all[idx].mean(axis=0)

        # ---------- 3. Pair‑wise Euclidean distances in C ------------------------
        dist_PO = distance.cdist(P_means, O_means, metric='euclidean')  # (n × n)
        dist_OO = distance.cdist(O_means, O_means, metric='euclidean')

        deltaD  = dist_PO - dist_OO
        np.fill_diagonal(deltaD, 0.0)            # ΔD[i,i] = 0 by definition
        return deltaD


    def change_direct_targets(self,adata, selected_idx, gene_input_dict):
        X = adata.X                   # must be CSR
        indptr, indices, data = X.indptr, X.indices, X.data
        # -------- resolve rows to int positions -------------
        # rows = np.asarray(
        #     adata.obs_names.get_indexer(selected_idx)
        #     if not np.issubdtype(np.asarray(selected_idx).dtype, np.integer)
        #     else selected_idx
        # )
        # rows = rows.astype(np.int64, copy=False)
        # -------- build factor vector -----------------------
        factors = np.ones(adata.n_vars, dtype=data.dtype)
        cols = adata.var_names.get_indexer(gene_input_dict.keys())
        factors = np.ones(adata.n_vars, dtype=data.dtype)
        factors[cols] = np.fromiter(
            (gene_input_dict[k] for k in gene_input_dict.keys()),
            dtype=data.dtype,
            count=len(gene_input_dict),
        )
        # --------- one‑line scale of every non‑zero ------------------
        data *= factors[indices]      # vectorised over the whole matrix
        return adata
    def perturb_input_data(self, selected_adata_index,perturb_resource, mode = 'direct'):
        data = ad.AnnData(X=self.data.X.copy())
        data.var_names = self.data.var.index.copy()
        data.obs.index = self.data.obs.index.copy()
        if type(perturb_resource) != dict:
            raise ValueError('perturb_resource should be a dict')
        if mode == 'direct':
            temp_data = self.change_direct_targets(data, selected_adata_index, perturb_resource)
        elif mode == 'GIN':
            temp_data = self.change_GIN_targets(data, selected_adata_index, perturb_resource)
        else:
            raise ValueError('mode should be in [\'direct\', \'GIN\']')
        self.data.obsm['perturbed_embedding'] = self._obtain_embedding(data = temp_data.X)
        
        
    def get_perturbed_embedding(self,selected_adata_index=None):
        if 'perturbed_embedding' not in self.data.obsm.keys():
            raise ValueError('No perturbed embedding found')
        if selected_adata_index is None:
            return self.data.obsm['perturbed_embedding']
        else:
            return self.data[selected_adata_index].obsm['perturbed_embedding']
    def calculate_similarity(self, target_data_index, selected_adata_index):
        target = self.data[target_data_index].obsm['embdedding']
        perturbed = self.data[selected_adata_index].obsm['perturbed_embedding']
        org_perturbed = self.data[selected_adata_index].obsm['embdedding'] 
        
        # calculate the mean embedding distance between perturbed and original and target
        target_perturbed = np.linalg.norm(np.mean(perturbed, axis = 0) - np.mean(target, axis = 0))

        target_org = np.linalg.norm(np.mean(org_perturbed, axis = 0) - np.mean(target, axis = 0))
        scores = self._calculateScore([target_perturbed-target_org, 0], flag=1)#-target_org
        return scores[0]

class perturbation:
    def __init__(self, data, model_name, idrem_dir, config_path=None):
        self.model_name = Path(model_name)

        self.idrem_dir = Path(idrem_dir)
        self.adata = data
        self.total_stage = len(set(self.adata.obs['stage']))
        self.tracks = self.getTrackReadOrder()
        self.stageadata = self.read_stagedata()
        
        
        
        self.hiddenReps = []
        self.perturb_stage_data_mean = []
        model_dir = self.model_name.parent
        if config_path is None:
            config_path = model_dir / 'training_parameters.json'

        self.pb = perturbator(model_path = self.model_name, data = self.adata, config_path = config_path)
    def read_stagedata(self):
        stageadata = []
        self.adata.obs['stage'] = self.adata.obs['stage'].astype('string')
        for i in list(self.adata.obs['stage'].unique()):

            stagedataids = self.adata.obs[self.adata.obs['stage']==i].index.values
            adata = self.adata[stagedataids]
            adata.obs['leiden'] = adata.obs['leiden'].astype('string')
            stageadata.append(adata)
        return stageadata
    def getTrackReadOrder(self):
        '''
        for each completed path in track (completed path = control->1->2->3, number of completed paths = number of 3 nodes), return a dictionary of orders. 
        like the path has stage3:1 is the second one to be read.
        '''
        path = self.idrem_dir
        filenames = os.listdir(path) #defalut path
        tempTrack = [[] for _ in range(self.total_stage)]
        for each in filenames:
            temp = each.split('.')[0].split('-')
            for i,item in enumerate(temp):
                temp1 = item.split('n')
                tempTrack[i].append(temp1)
        dic = {}
        for i, ids in enumerate(tempTrack[-1]):
            for each in ids:
                dic[int(each)] = i
        return dic
    def getTrack(self,stage, clusterid):
        path = self.idrem_dir
        filenames = os.listdir(path) #defalut path

        tempTrack = [[] for _ in range(self.total_stage)]
        for each in filenames:
            temp = each.split('.')[0].split('-')
            for i,item in enumerate(temp):
                temp1 = item.split('n')
                tempTrack[i].append(temp1)
        track = [[] for _ in range(self.total_stage)]

        edges = self.adata.uns['edges']
        for i, each in enumerate(tempTrack[int(stage)]):
            if str(clusterid) in each:
                track[0] = [int(tempTrack[0][i][0])]
                
                tempcluster = clusterid
                for k in range(int(stage),0, -1):
                    for new_each in edges[str(k-1)]:
                        if new_each[1] == tempcluster:
                            track[k]= [new_each[1]]
                            tempcluster = new_each[0] 
                            break
                    
                tempcluster = [clusterid]
                
                for k in range(int(stage)+1,self.total_stage):
                    
                    track[k]=self.getDescendants(tempcluster,k,edges)
                    tempcluster=track[k]
        return track
    def run(self,mode,log2fc,inplace=False,random_times = 100,random_genes = None,CUDA = False,device = 'cuda:0',GIN=False):
        '''
        Perform perturbation.

        parameters
        -------------------
        mode: str
            perturbation mode, 'drug', 'pathway', 'random_background', 'online_random_background', 'single_gene', 'two_genes'
        log2fc: float
            log2fc of the perturbation
        inplace: bool
            whether to write the perturbation results to the adata object
        random_times: int
            number of random genes to be perturbed
        random_genes: int
            number of random genes to be perturbed
        CUDA: bool
            whether to use CUDA
        device: str
            device to be used, 'cuda:0' or 'cpu'
        GIN: bool
            whether to use Gene-Gene Interaction Network (GIN) for perturbation
        return
        -------------------
        None
        '''
        if inplace == False:
            written=True
        else:
            written=False
        
        # --- PRECOMPUTE dictionaries -----------------------------------------
        stage_leiden2rows = {}
        for stage in self.adata.obs['stage'].unique():
            for leiden in self.adata.obs['leiden'].unique():
                mask = (self.adata.obs['stage'] == stage) & (self.adata.obs['leiden'] == leiden)
                if len(self.adata.obs.index[mask]) == 0:
                    continue
                stage_leiden2rows[(stage, leiden)] = self.adata.obs.index[mask].to_numpy()
        track_cache = {}
        for lastCluster, full_track in self.tracks.items():
            track = self.getTrack(len(self.stageadata)-1,lastCluster)
            track_name = str(track[0][0])
            for i in range(1,len(track)):
                track_name += '-' + str(track[i][0])

            # key = tuple(track[0] for track in self.getTrack(len(self.stageadata)-1,lastCluster))   # your old logic
            rows = [np.concatenate([stage_leiden2rows[(str(s), str(l))]
                                    for l in clusters])
                    for s, clusters in enumerate(track)]
            track_cache[track_name] = rows      # reuse later
        # --- END PRECOMPUTE ---------------------------------------------------
        if mode == 'drug':
            drug_gene = self.adata.uns['data_drug_overlap_genes']
            perturbed_items = list(drug_gene.keys())
            temp_perturbed_genes = list(drug_gene.values())
            perturbed_genes = []
            for each_direction in [log2fc,1/log2fc]:
                for perturbation_item_idx, genes in tqdm(enumerate(temp_perturbed_genes),total = len(temp_perturbed_genes)):
                    gene_input_dict = {}
                    if type(genes)!= list:
                        genes = genes.tolist()
                    
                    plus  = each_direction          # cache the two possible results
                    minus = 1 / each_direction

                    for item in genes:              # e.g. "NAT2:-"
                        name, _, sign = item.partition(':')   # one C‑level split
                        gene_input_dict[name] = plus if sign == '+' else minus

                    perturb_cells = self.adata.obs.index
                    import time
                    self.pb.perturb_input_data(perturb_cells,gene_input_dict,mode='direct')
                    if 'drug_perturbation_deltaD' not in self.adata.uns.keys():
                        self.adata.uns['drug_perturbation_deltaD'] = {}
                    if str(each_direction) not in self.adata.uns['drug_perturbation_deltaD'].keys():
                        self.adata.uns['drug_perturbation_deltaD'][str(each_direction)] = {}
                    for lastCluster in self.tracks.keys():
                        
                        track_indices = []
                        track = self.getTrack(len(self.stageadata)-1,lastCluster)
                        track_name = str(track[0][0]) 
                        for i in range(1,len(track)):
                            track_name += '-' + str(track[i][0])
                        track_indices = track_cache[track_name]
                        deltaD = self.pb.calculate_deltaD_for_tracks(track_indices)
                        
                        if track_name not in self.adata.uns['drug_perturbation_deltaD'][str(each_direction)].keys():
                            self.adata.uns['drug_perturbation_deltaD'][str(each_direction)][track_name] = {}
                        self.adata.uns['drug_perturbation_deltaD'][str(each_direction)][track_name][perturbed_items[perturbation_item_idx]] = deltaD
                    del self.pb.data.obsm['perturbed_embedding']
                    gc.collect() # collect garbage to free memory
            print('finished')

        elif mode == 'pathway':
            pathway_gene = self.adata.uns['data_pathway_overlap_genes']
            perturbed_items = list(pathway_gene.keys())
            temp_perturbed_genes = list(pathway_gene.values())
            for each_direction in [log2fc,1/log2fc]:
                for perturbation_item_idx, genes in tqdm(enumerate(temp_perturbed_genes),total = len(temp_perturbed_genes)):
                    gene_input_dict = {}
                    if type(genes)!= list:
                        genes = genes.tolist()
                    # for each in genes:
                    #     gene_input_dict[each] = each_direction
                    gene_input_dict = dict.fromkeys(genes, each_direction) 
    
                    perturb_cells = self.adata.obs.index
                    self.pb.perturb_input_data(perturb_cells,gene_input_dict,mode='direct')
                    if 'pathway_perturbation_deltaD' not in self.adata.uns.keys():
                        self.adata.uns['pathway_perturbation_deltaD'] = {}
                    if str(each_direction) not in self.adata.uns['pathway_perturbation_deltaD'].keys():
                        self.adata.uns['pathway_perturbation_deltaD'][str(each_direction)] = {}
                    for lastCluster in self.tracks.keys():
                        track_indices = []
                        track = self.getTrack(len(self.stageadata)-1,lastCluster)
                        track_name = str(track[0][0]) 
                        for i in range(1,len(track)):
                            track_name += '-' + str(track[i][0])
                        track_indices = track_cache[track_name]

                        deltaD = self.pb.calculate_deltaD_for_tracks(track_indices)
                        if track_name not in self.adata.uns['pathway_perturbation_deltaD'][str(each_direction)].keys():
                            self.adata.uns['pathway_perturbation_deltaD'][str(each_direction)][track_name] = {}
                        self.adata.uns['pathway_perturbation_deltaD'][str(each_direction)][track_name][perturbed_items[perturbation_item_idx]] = deltaD
                    del self.pb.data.obsm['perturbed_embedding']
                    gc.collect() # collect garbage to free memory
            print('finished')
        elif mode == 'random_pathway_background':
            pathway_gene = self.adata.uns['data_pathway_overlap_genes']
            perturbed_items = list(pathway_gene.keys())
            temp_perturbed_genes = list(pathway_gene.values())
            perturbed_genes = []
            for each_direction in [log2fc,1/log2fc]:
                for perturbation_item_idx, genes in tqdm(enumerate(temp_perturbed_genes),total = len(temp_perturbed_genes)):
                    gene_input_dict = {}
                    if type(genes)!= list:
                        genes = genes.tolist()
                    genes = get_random_genes(self.adata, len(genes), perturbation_item_idx) #use random genes
                    gene_input_dict = dict.fromkeys(genes, each_direction) 

                    perturb_cells = self.adata.obs.index
                    self.pb.perturb_input_data(perturb_cells,gene_input_dict,mode='direct')
                    
                    if 'random_background_perturbation_deltaD' not in self.adata.uns.keys():
                        self.adata.uns['random_background_perturbation_deltaD'] = {}
                    if str(each_direction) not in self.adata.uns['random_background_perturbation_deltaD'].keys():
                        self.adata.uns['random_background_perturbation_deltaD'][str(each_direction)] = {}
                    for lastCluster in self.tracks.keys():
                        track_indices = []
                        track = self.getTrack(len(self.stageadata)-1,lastCluster)
                        track_name = str(track[0][0]) 
                        for i in range(1,len(track)):
                            track_name += '-' + str(track[i][0])
                        track_indices = track_cache[track_name]
                        deltaD = self.pb.calculate_deltaD_for_tracks(track_indices)
                        if track_name not in self.adata.uns['random_background_perturbation_deltaD'][str(each_direction)].keys():
                            self.adata.uns['random_background_perturbation_deltaD'][str(each_direction)][track_name] = {}
                        self.adata.uns['random_background_perturbation_deltaD'][str(each_direction)][track_name][perturbed_items[perturbation_item_idx]] = deltaD
                    del self.pb.data.obsm['perturbed_embedding']
                    gc.collect() # collect garbage to free memory
            print('finished')
        elif mode == 'random_drug_background':
            if 'random_background_perturbation_deltaD' in self.adata.uns.keys():
                self.adata.uns['random_background_perturbation_deltaD'] = {}
            drug_gene = self.adata.uns['data_drug_overlap_genes']
            perturbed_items = list(drug_gene.keys())
            temp_perturbed_genes = list(drug_gene.values())
            perturbed_genes = []
            for each_direction in [log2fc,1/log2fc]:
                for perturbation_item_idx, genes in tqdm(enumerate(temp_perturbed_genes),total = len(temp_perturbed_genes)):
                    if perturbation_item_idx > random_times:
                        print('Random drug background perturbation stopped as reaching the set random times:', random_times)
                        break
                    gene_input_dict = {}
                    if type(genes)!= list:
                        genes = genes.tolist()
                    if random_genes is not None:
                        genes = get_random_genes(self.adata, random_genes, perturbation_item_idx) #use random genes, # of genes = median number of genes in the customized drug profile
                    else:
                        genes = get_random_genes(self.adata, len(genes), perturbation_item_idx) #use random genes
                    gene_input_dict = dict.fromkeys(genes, each_direction) 

                    perturb_cells = self.adata.obs.index
                    self.pb.perturb_input_data(perturb_cells,gene_input_dict,mode='direct')
                    
                    if 'random_background_perturbation_deltaD' not in self.adata.uns.keys():
                        self.adata.uns['random_background_perturbation_deltaD'] = {}
                    if str(each_direction) not in self.adata.uns['random_background_perturbation_deltaD'].keys():
                        self.adata.uns['random_background_perturbation_deltaD'][str(each_direction)] = {}
                    for lastCluster in self.tracks.keys():
                        track_indices = []
                        track = self.getTrack(len(self.stageadata)-1,lastCluster)
                        track_name = str(track[0][0]) 
                        for i in range(1,len(track)):
                            track_name += '-' + str(track[i][0])
                        track_indices = track_cache[track_name]
                        deltaD = self.pb.calculate_deltaD_for_tracks(track_indices)
                        if track_name not in self.adata.uns['random_background_perturbation_deltaD'][str(each_direction)].keys():
                            self.adata.uns['random_background_perturbation_deltaD'][str(each_direction)][track_name] = {}
                        self.adata.uns['random_background_perturbation_deltaD'][str(each_direction)][track_name][perturbed_items[perturbation_item_idx]] = deltaD
                    del self.pb.data.obsm['perturbed_embedding']
                    gc.collect() # collect garbage to free memory
        elif mode == 'single_gene':
            all_genes = self.adata.var_names.tolist()
            all_genes = {each: [each] for each in all_genes}
            perturbed_items = list(all_genes.keys())
            temp_perturbed_genes = list(all_genes.values())
            for each_direction in [log2fc,1/log2fc]:
                for perturbation_item_idx, genes in tqdm(enumerate(temp_perturbed_genes),total = len(temp_perturbed_genes)):
                    gene_input_dict = {}
                    if type(genes)!= list:
                        genes = genes.tolist()
                    # for each in genes:
                    #     gene_input_dict[each] = each_direction
                    gene_input_dict = dict.fromkeys(genes, each_direction) 
    
                    perturb_cells = self.adata.obs.index
                    self.pb.perturb_input_data(perturb_cells,gene_input_dict,mode='direct')
                    if 'single_gene_perturbation_deltaD' not in self.adata.uns.keys():
                        self.adata.uns['single_gene_perturbation_deltaD'] = {}
                    if str(each_direction) not in self.adata.uns['single_gene_perturbation_deltaD'].keys():
                        self.adata.uns['single_gene_perturbation_deltaD'][str(each_direction)] = {}
                    for lastCluster in self.tracks.keys():
                        track_indices = []
                        track = self.getTrack(len(self.stageadata)-1,lastCluster)
                        track_name = str(track[0][0]) 
                        for i in range(1,len(track)):
                            track_name += '-' + str(track[i][0])
                        track_indices = track_cache[track_name]

                        deltaD = self.pb.calculate_deltaD_for_tracks(track_indices)
                        if track_name not in self.adata.uns['single_gene_perturbation_deltaD'][str(each_direction)].keys():
                            self.adata.uns['single_gene_perturbation_deltaD'][str(each_direction)][track_name] = {}
                        self.adata.uns['single_gene_perturbation_deltaD'][str(each_direction)][track_name][perturbed_items[perturbation_item_idx]] = deltaD
                    del self.pb.data.obsm['perturbed_embedding']
                    gc.collect() # collect garbage to free memory
            print('Single gene perturbation finished!')
        elif mode =='two_genes':
            all_genes = self.adata.var_names.tolist()
            combination_gene_set1 = self.adata.uns['combinatorial_perturbation_genes_set1']
            candidate_dict = {}
            for each in combination_gene_set1:
                for gene in all_genes:
                    if gene != each:
                        candidate_dict[each+'_'+gene] = []
                        candidate_dict[each+'_'+gene].append(each)
                        candidate_dict[each+'_'+gene].append(gene)

            perturbed_items = list(candidate_dict.keys())
            temp_perturbed_genes = list(candidate_dict.values())
            for each_direction in [log2fc,1/log2fc]:
                for perturbation_item_idx, genes in tqdm(enumerate(temp_perturbed_genes),total = len(temp_perturbed_genes)):
                    gene_input_dict = {}
                    if type(genes)!= list:
                        genes = genes.tolist()
                    # for each in genes:
                    #     gene_input_dict[each] = each_direction
                    gene_input_dict = dict.fromkeys(genes, each_direction) 
    
                    perturb_cells = self.adata.obs.index
                    self.pb.perturb_input_data(perturb_cells,gene_input_dict,mode='direct')
                    if 'two_genes_perturbation_deltaD' not in self.adata.uns.keys():
                        self.adata.uns['two_genes_perturbation_deltaD'] = {}
                    if str(each_direction) not in self.adata.uns['two_genes_perturbation_deltaD'].keys():
                        self.adata.uns['two_genes_perturbation_deltaD'][str(each_direction)] = {}
                    for lastCluster in self.tracks.keys():
                        track_indices = []
                        track = self.getTrack(len(self.stageadata)-1,lastCluster)
                        track_name = str(track[0][0]) 
                        for i in range(1,len(track)):
                            track_name += '-' + str(track[i][0])
                        track_indices = track_cache[track_name]

                        deltaD = self.pb.calculate_deltaD_for_tracks(track_indices)
                        if track_name not in self.adata.uns['two_genes_perturbation_deltaD'][str(each_direction)].keys():
                            self.adata.uns['two_genes_perturbation_deltaD'][str(each_direction)][track_name] = {}
                        self.adata.uns['two_genes_perturbation_deltaD'][str(each_direction)][track_name][perturbed_items[perturbation_item_idx]] = deltaD
                    del self.pb.data.obsm['perturbed_embedding']
                    gc.collect() # collect garbage to free memory
        elif mode == 'random_background':
            for each_direction in [log2fc,1/log2fc]:
                for perturbation_item_idx in tqdm(range(random_times),total = random_times):
                    gene_input_dict = {}
                    genes = get_random_genes(self.adata, random_genes, perturbation_item_idx) #use random genes
                    gene_input_dict = dict.fromkeys(genes, each_direction) 

                    perturb_cells = self.adata.obs.index
                    self.pb.perturb_input_data(perturb_cells,gene_input_dict,mode='direct')
                    
                    if 'random_background_perturbation_deltaD' not in self.adata.uns.keys():
                        self.adata.uns['random_background_perturbation_deltaD'] = {}
                    if str(each_direction) not in self.adata.uns['random_background_perturbation_deltaD'].keys():
                        self.adata.uns['random_background_perturbation_deltaD'][str(each_direction)] = {}
                    for lastCluster in self.tracks.keys():
                        track_indices = []
                        track = self.getTrack(len(self.stageadata)-1,lastCluster)
                        track_name = str(track[0][0]) 
                        for i in range(1,len(track)):
                            track_name += '-' + str(track[i][0])
                        track_indices = track_cache[track_name]
                        deltaD = self.pb.calculate_deltaD_for_tracks(track_indices)
                        if track_name not in self.adata.uns['random_background_perturbation_deltaD'][str(each_direction)].keys():
                            self.adata.uns['random_background_perturbation_deltaD'][str(each_direction)][track_name] = {}
                        self.adata.uns['random_background_perturbation_deltaD'][str(each_direction)][track_name][perturbation_item_idx] = deltaD
                    del self.pb.data.obsm['perturbed_embedding']
                    gc.collect() # collect garbage to free memory
            print('finished')
    def analysis(self,mode,log2fc,perturbed_tracks='all',overall_perturbation_analysis=True,stage=None):
        '''
        Analysis of perturbation results
        
        parameters
        ----------------
        mode: str
            The mode is choosing from ['drug', 'pathway', 'online','single_gene','two_genes']
        log2fc: float
            log2fc is the log2 fold change of perturbation
        overall_perturbation_analysis: bool
            overall_perturbation_analysis is whether to calculate perturbation score for all tracks as a whole or individually. True: all tracks. False: one track.
        stage: int
            stage is the stage to be analysis, if all is True, stage is None
        '''

        self.adata.obs['leiden'] = self.adata.obs['leiden'].astype('string')
        self.adata.obs['stage'] = self.adata.obs['stage'].astype('string')
        analyst = perturbationAnalysis(self.adata,self.idrem_dir, stage=stage,log2fc = log2fc, mode = mode)
       
        
        temp = analyst.main_analysis(perturbed_tracks = perturbed_tracks, overall_perturbation_analysis=overall_perturbation_analysis, score='avg_backScore', items=None)#read item from disk for now
        if '%s_perturbation_score'%mode not in self.adata.uns.keys():
            self.adata.uns['%s_perturbation_score'%mode] = {}
        
        self.adata.uns['%s_perturbation_score'%mode][str(log2fc)] = temp
        self.adata.uns['%s_perturbation_score'%mode][str(1/log2fc)] = temp
        self.adata.obs['stage'] = self.adata.obs['stage'].astype(str)