import scanpy as sc
import torch
from torch.utils.data import DataLoader
from UNAGI.utils.gcn_utils import setup_graph
from UNAGI.model.models import VAE
from scipy.sparse import issparse
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
def get_overlap_genes_pathway(adata, pathway):
    genes = adata.var.index
    overlap_genes = []
    for each in pathway:
        each = each.upper()
        if each in genes:
            overlap_genes.append(each)
    return overlap_genes
class perturbator:
    def __init__(self, model_path,data_path,config_path):
        torch.load(model_path)
        self.data = sc.read(data_path)
        self.data.var_names_make_unique()
        self.config = json.load(open(config_path))
        self.model = VAE(self.config['input_dim'], self.config['hidden_dim'], self.config['graph_dim'], self.config['latent_dim'],beta=self.config['beta'],distribution=self.config['dist'])
        self.model.load_state_dict(torch.load(model_path,map_location=torch.device(self.config['device'])))
        self.model.to(self.config['device'])
        self.model.eval()
        self.data.obsm['embdedding'] = self._obtain_embedding()
    def _obtain_embedding(self, data = None):
        adj = self.data.obsp['gcn_connectivities']
        if data is None:
            data = self.data.X
        if issparse(data):
            data = data.toarray()
        cell_loader = DataLoader(data.astype('float32'), batch_size=len(data), shuffle=False, num_workers=0)
        adj = adj.asformat('coo')
        adj = setup_graph(adj)
        adj = adj.to('cuda:0')
        for perturbed_index, x in enumerate(cell_loader):
            x = x.to('cuda:0')
            z = self.model.get_latent_representation(x,adj)
        # z to np array
        z = z.detach().cpu().numpy()
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
    def getPPINetworkDict(self, adata, hippiefile = './hippie_current.txt',stringfile = '9606.protein.links.v11.5.txt',stringinfo = '9606.protein.info.v11.5.txt', output_GIN_path = 'PPINetworkDict.npy'):
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
    def preprocess_GIN(self, selected_adata_index, perturb_resource, GIN_path = './PPINetworkDict.npy'):
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
    
    def change_GIN_targets(self, adata,selected_adata_index, perturb_resource,GIN_path='./PPINetworkDict.npy'):
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
    def change_direct_targets(self, adata,selected_adata_index, perturb_resource):
        for each_gene in perturb_resource:
            adata[selected_adata_index,each_gene].X = adata[selected_adata_index,each_gene].X * perturb_resource[each_gene]
        return adata
    def perturb_input_data(self, selected_adata_index,perturb_resource, mode = 'direct'):
        if type(perturb_resource) != dict:
            raise ValueError('perturb_resource should be a dict')
        temp_data = self.data.copy()
        if mode == 'direct':
            temp_data = self.change_direct_targets(temp_data, selected_adata_index, perturb_resource)
        elif mode == 'GIN':
            temp_data = self.change_GIN_targets(temp_data, selected_adata_index, perturb_resource)
        else:
            raise ValueError('mode should be in [\'direct\', \'GIN\']')
        z = self._obtain_embedding(data = temp_data.X)
        self.data.obsm['perturbed_embedding'] = z
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
def get_random_genes(adata, n,seed):
    np.random.seed(seed)
    genes = adata.var.index
    return np.random.choice(genes,n)

if __name__ == '__main__':
    RANDOM_BACKGROUND = True
    import argparse
    import json
    parser = argparse.ArgumentParser(description='Run perturbation analysis')
    parser.add_argument('--MODEL_PATH', type=str, help='model path')
    parser.add_argument('--DATA_PATH', type=str, help='path of the adata file')
    parser.add_argument('--MODEL_CONF', type=str, help='the path of the model config file')
    parser.add_argument('--PERTURB_CONF', type=str, help='the path of the perturbation config file')
    perturbation_conf = json.load(open(args.PERTURB_CONF))
    args = parser.parse_args()
    # MODEL_PATH = '/mnt/md0/jiahui/sex_chimeric/WT_mdx/data/model_save/WT_mdx_2.pth'
    # DATA_PATH = '/mnt/md0/jiahui/sex_chimeric/WT_mdx/data/2/stagedata/dataset.h5ad'
    # CONFIG_PATH = '/mnt/md0/jiahui/sex_chimeric/WT_mdx/data/model_save/training_parameters.json'
    df = {}
    pb = perturbator(args.MODEL_PATH, args.DATA_PATH, args.CONFIG_PATH)
    subset_adata = pb.data
    subset_adata = subset_adata[subset_adata.obs[perturbation_conf.identification]=='WT_mdx']
    perturb_cells = subset_adata.obs.index
    a = combined_pathways

    #non_lung_cells_adata_idx is your stage 0, lung_cells_adata_idx is your stage 1.
    lung_cells_adata = pb.data[pb.data.obs[perturbation_conf.identification].isin(['WT_WT'])]
    lung_cells_adata_idx = lung_cells_adata.obs.index



    for pathway in tqdm(list(a.keys()),desc='perturbating pathways...'):
        genes = get_overlap_genes_pathway(subset_adata,a[pathway])
        if RANDOM_BACKGROUND:
            genes = get_random_genes(subset_adata, len(genes))
        gene_input_dict_negative = {}
        gene_input_dict_positive = {}
        for each in genes:
            gene_input_dict_negative[each] = 0.5
            gene_input_dict_positive[each] = 2
        
        
        pb.perturb_input_data(perturb_cells,gene_input_dict_positive,mode='direct')
        positive_perturbation_score = pb.calculate_similarity(lung_cells_adata_idx, perturb_cells)

        #---------#
        pb.perturb_input_data(perturb_cells,gene_input_dict_negative,mode='direct')
        negative_perturbation_score = pb.calculate_similarity(lung_cells_adata_idx, perturb_cells)

        #---------#
        final_score = np.abs(np.mean(positive_perturbation_score-negative_perturbation_score))
        df[pathway] = final_score/2
