#this is push back, last cell is push forward
import numpy as np
import os
import random
import gc
import torch
from scipy.sparse import issparse
#import DataLoader from torch
from torch.utils.data import DataLoader
from ..utils.gcn_utils import setup_graph
import threading
from ..model.models import VAE
from .analysis_perturbation import perturbationAnalysis
class perturbation:
    def __init__(self, target_directory,model_name,idrem_dir):
        self.model_name = model_name
        self.target_directory = target_directory
        
        self.idrem_dir = idrem_dir
        

        self.adata = self.read_mergeadata()
        self.total_stage = len(set(self.adata.obs['stage']))
        self.tracks = self.getTrackReadOrder()
        self.stageadata = self.read_stagedata()
        
        
        self.hiddenReps = []
        self.perturb_stage_data_mean = []

        
    def read_mergeadata(self):
        read_path = self.target_directory#os.path.join(self.target_directory,'dataset.h5ad')
        mergeadata = read_path#sc.read_h5ad(read_path)
        mergeadata.obs['leiden'] = mergeadata.obs['leiden'].astype('string')
        return mergeadata
    def read_stagedata(self):
        stageadata = []
        self.stage_cluster = {}
        stage_have_clusters = {}
        stage_have_clusters[str(0)] = []
        for i in self.tracks.keys():
            track = self.getTrack(len(self.adata.obs['stage'].unique())-1,i)
            track_name = str(track[0][0])
            stage_have_clusters[str(0)].append(str(track[0][0]))
            for j in range(1,len(track)):
                if str(j) not in stage_have_clusters.keys():
                    stage_have_clusters[str(j)] = []
                stage_have_clusters[str(j)].append(str(track[j][0]))
                track_name += '-' + str(track[j][0])
        self.adata.obs['stage'] = self.adata.obs['stage'].astype('string')
        for i in range(0,4):

            stagedataids = self.adata.obs[self.adata.obs['stage']==str(i)].index.values
            

            adata = self.adata[stagedataids]
            adata.obs['leiden'] = adata.obs['leiden'].astype('string')
            grouped = adata.obs.groupby('leiden')
            self.stage_cluster[str(i)] = {}#{name: adata[group.index.tolist()] for name, group in grouped}
            for name, group in grouped:
                if str(name) not in stage_have_clusters[str(i)]:
                    continue
                else:
                    self.stage_cluster[str(i)][str(name)] = adata[group.index.tolist()]
            stageadata.append(adata)
        return stageadata
    def get_gene_names(self):
        return np.array(list(self.stageadata[0].var.index.values))
    # def save_to_disk(self):
    #         for i, each in enumerate(self.stageadata):
    #             save_path = os.path.join(self.target_directory,'stagedata/concat_%d.h5ad'%(i))
    #             each.write_h5ad(save_path)
    def getDescendants(tempcluster,stage,edges):
        out = []
        for each in tempcluster:
            
            for item in edges[str(stage-1)]:
            
                if each == item[0]:
                    
                    out.append(item[1])
        return out
    def getDistance(self,rep, cluster):
        #use centroid to represent the cluster
        cluster = np.mean(cluster,axis=0)
        if rep.ndim == 2:
            rep = np.mean(rep,axis=0)
            return np.linalg.norm(rep-cluster)
        elif rep.ndim == 3:
            rep = np.mean(rep,axis=1)
            return np.linalg.norm(rep-cluster,axis=1)
        else:
            raise ValueError('rep should be 2 or 3 dimension')

        
        # cluster = cluster.reshape(1,-1)
        # cluster = cluster.repeat(rep.shape[0],axis=0)
        
    def matchSingleClusterGeneDict(self,goadata,gotop):
        gene_dict={}
        for i,each in enumerate(goadata.var.index.tolist()):
            gene_dict[each]=i
        results=[]
        for each in gotop:
            if each not in goadata.var.index.tolist():

                continue
            results.append(gene_dict[each])
        return results
    def getTrack(self,stage, clusterid):
        path = self.idrem_dir#os.path.join(self.target_directory,'idremVizCluster')
        filenames = os.listdir(path) #defalut path

        tempTrack = [[] for _ in range(self.total_stage)]
        for each in filenames:
            temp = each.split('.')[0].split('-')
            for i,item in enumerate(temp):
                temp1 = item.split('n')
                tempTrack[i].append(temp1)
        track = [[] for _ in range(self.total_stage)]
        # open_path = os.path.join(self.target_directory,'edges.txt')
        # edges = eval(open(open_path, 'r').read())
        edges = self.adata.uns['edges']
        for i, each in enumerate(tempTrack[int(stage)]):
            if str(clusterid) in each:
                track[0] = [int(tempTrack[0][i][0])]
                
                tempcluster = clusterid
                for k in range(int(stage),0, -1):
                    for new_each in edges[str(k-1)]:
                        if new_each[1] == tempcluster:
                            track[k]= [new_each[1]]#[edges[str(k-1)][tempcluster][1]]
                            tempcluster = new_each[0] #edges[str(k-1)][tempcluster][0] 
                            break
                    # track[k]= [edges[str(k-1)][tempcluster][1]]
                    
                tempcluster = [clusterid]
                
                for k in range(int(stage)+1,self.total_stage):
                    
                    track[k]=self.getDescendants(tempcluster,k,edges)
                    tempcluster=track[k]
        return track

    def getZandZc(self,adata,stage,cluster,CUDA = False, impactfactor=None,topN = None):
        if stage < len(self.hiddenReps):
            if impactfactor is None:
                return self.hiddenReps[stage]
            else:
                data,adj = self.perturb_stage_data_mean[stage]
        else:
            
            

            clusterAdataID = adata.obs[adata.obs['leiden'] == str(cluster)].index.tolist()
            clusterAdata = adata[clusterAdataID]
            adj = clusterAdata.obsp['gcn_connectivities']
            # gcn_connectivities = clusterAdata.obsp['gcn_connectivities']
            
            # temp = gcn_connectivities[clusterAdataID]
            # temp = list(set(temp.indices))
            # # input = np.zeros(shape = (adata.X.shape[0],adata.X.shape[1]))
            # input = csr_matrix(input)
                
            # input[temp] = adata.X[temp]
            # input = gcn_connectivities @ input
            # input = input[temp].toarray()
            data = clusterAdata.X#clusterAdata.obsp['gcn_connectivities'] @ clusterAdata.X
            # if issparse(input):
                # input = input.toarray()
            
            # data = np.mean(input,axis=0).reshape(1,-1)
        loadModelDict = self.model_name#'./'+self.target_directory+'/model_save/'+self.model_name+'.pth'
        vae = VAE(data.shape[1], 256, 1024, 64,beta=1,distribution='ziln')#torch.load(loadModelDict)
        # vae = torch.load(loadModelDict)nput_dim, hidden_dim, graph_dim, latent_dim
        # vae = VAE(data.shape[1], 64, 256, 0.5)
        if CUDA:
            vae.load_state_dict(torch.load(loadModelDict,map_location=torch.device('cuda:0')))
            # vae = torch.load(loadModelDict)
            vae.to('cuda:0')
        else:

            vae.load_state_dict(torch.load(loadModelDict), map_location=torch.device('cpu'))
            vae.to('cpu')
        
        vae.eval()
        recons = []
        zs = []
        zmeans = []
        zstds = []
        # data = np.mean(clusterAdata.layers['concat'].toarray(),axis=0).reshape(1,-1)#
        # print(np.mean(data))
        if impactfactor is not None:
            data = np.expand_dims(data,axis=0)
            data = np.repeat(data, len(impactfactor), axis=0)
            print(data.shape)
            cell_loader = DataLoader(data.astype('float32'), batch_size=1, shuffle=False, num_workers=0)
        else:
            data = data.toarray()
            cell_loader = DataLoader(data.astype('float32'), batch_size=len(data), shuffle=False, num_workers=0)
            adj = adj.asformat('coo')
            adj = setup_graph(adj)
        # cell_loader = DataLoader(clusterAdata.layers['concat'].toarray().astype('float32'), batch_size=2000, shuffle=False, num_workers=1)
        # xx_adj = np.ones(np.shape(clusterAdata.X)[0])
        
        # adj = torch.tensor(1)
    #if it's not tensor then as format

        
        if CUDA:
            adj = adj.to('cuda:0')
        for perturbed_index, x in enumerate(cell_loader):
            
            if impactfactor is not None:
                x = x.squeeze(0)
                impactfactor = impactfactor.astype('float32')
                # x = x+impactfactor

                x = x+x*impactfactor[perturbed_index]
            
    
            # if on GPU put mini-batch into CUDA memory
            if CUDA:
                x = x.to('cuda:0')
            z = vae.get_latent_representation(x,adj)
            
            zs+=z.cpu().detach().numpy().tolist()
           
        zs = np.array(zs)
        if impactfactor is not None:
            zs = zs.reshape(len(impactfactor),-1,64)
        print('zs.shape',zs.shape)
        if stage >= len(self.hiddenReps):
            self.hiddenReps.append(zs)
            self.perturb_stage_data_mean.append([data,adj])
        return zs
    class perturbationthread(threading.Thread):
        def __init__(self, outer_instance,outs, selectedstage,selectedcluster,track,bound,perturbated_gene,CUDA):
            threading.Thread.__init__(self)
            self.selectedstage = selectedstage
            self.selectedcluster = selectedcluster
            self.track = track
            self.bound = bound
            self.outs = outs
            self.perturbated_gene = perturbated_gene
            self.outer_instance = outer_instance
            self.CUDA = CUDA
        def run(self):
            self.outs[self.selectedstage]+=self.outer_instance.perturbation__auto_centroid(self.outer_instance.stageadata[self.selectedstage], self.outer_instance.stageadata, self.selectedstage, self.selectedcluster, self.track, self.bound,self.perturbated_gene,self.CUDA)

    def getTrackReadOrder(self):
        '''
        for each completed path in track (completed path = control->1->2->3, number of completed paths = number of 3 nodes), return a dictionary of orders. 
        like the path has stage3:1 is the second one to be read.
        '''
        path = self.idrem_dir#os.path.join(self.target_directory,'idremVizCluster')
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
    def perfect_perturbation__auto_centroid(self,stageadata, adata, selectedstage,selectedcluster,track,bound,perturbated_genes,CUDA=False):
        '''
        remove non top genes and tf. compared
        '''

        hiddenReps=[]
        repNodes = []
        flag = -1 #to indicate which cluster in the track to be changed like [40,1,3,2,4,5] 3 is no.2
        clusterids = []
        zc = []
        adatacollection = adata#[]
        
        plotadata = []
        for stage, clusters in enumerate(track):
            for clusterid, leiden in enumerate(clusters):
                if stage == selectedstage and leiden == selectedcluster:
                    flag = len(hiddenReps)
                
                temp = self.getZandZc(adatacollection[stage],stage,leiden,CUDA=CUDA)
            
                hiddenReps.append(temp)
        hiddenReps = np.array(hiddenReps)

        dijresults = []
        
        count=0
        
        for stage, clusters in enumerate(track):
            temp = []
            for clusterid, leiden in enumerate(clusters):
            
                temp=self.getDistance(hiddenReps[flag],hiddenReps[count])
                count+=1
            dijresults.append(temp)
            

        perturbated_stage = int(selectedstage)
  
        
        
        impactFactor = perturbated_genes
        impactFactor[impactFactor!=0] = 1
        temp = impactFactor[0].copy()
        perturbated_stage = int(selectedstage)
        selectedstage=1
        if bound > 0:
            mean_previous = np.mean(np.array(adatacollection[selectedstage-1].X.toarray()),axis=0)
            mean_current = np.mean(np.array(adatacollection[selectedstage].X.toarray()),axis=0)
            if selectedstage == 0:
                mean_previous = mean_current
            
            diff = mean_previous - mean_current
            por = diff/mean_current
            impactFactor = por * impactFactor
        else:

            mean_previous = np.mean(np.array(adatacollection[selectedstage+1].X.toarray()),axis=0)
            mean_current = np.mean(np.array(adatacollection[selectedstage].X.toarray()),axis=0)
            if selectedstage == len(adata)-1:
                mean_previous = mean_current
            diff = mean_previous - mean_current
            por = diff/mean_current
            impactFactor = por * impactFactor

        selectedtemp = self.getZandZc(None,selectedstage,track[selectedstage][0],impactfactor = impactFactor,CUDA=CUDA)
        count = 0
        fijresults = []
        for stage, clusters in enumerate(track):
            temp = []
            for clusterid, leiden in enumerate(clusters):
                temp = self.getDistance(selectedtemp,hiddenReps[count])
                count+=1
            fijresults.append(temp)
        delta = np.array(fijresults) - np.array(dijresults)

        gc.collect()
        
        out = []
        for i in range(impactFactor.shape[0]):
            temp = []
            temp.append(perturbated_genes[i])
            for kk in range(len(track)):
                temp.append(track[kk][0])
            for kk in range(len(track)):
                temp.append(delta[kk][i])
            out.append(temp)

        return out
    def perturbation__auto_centroid(self,stageadata, adata, selectedstage,selectedcluster,track,bound,perturbated_genes,CUDA=False):
        '''
        remove non top genes and tf. compared
        '''

        hiddenReps=[]
        repNodes = []
        flag = -1 #to indicate which cluster in the track to be changed like [40,1,3,2,4,5] 3 is no.2
        clusterids = []
        zc = []
        adatacollection = adata#[]
        
        plotadata = []
        for stage, clusters in enumerate(track):
            for clusterid, leiden in enumerate(clusters):
                if stage == selectedstage and leiden == selectedcluster:
                    flag = len(hiddenReps)
               
                temp = self.getZandZc(adatacollection[stage],stage,leiden,CUDA=CUDA)
            
                hiddenReps.append(temp)
        hiddenReps = np.array(hiddenReps, dtype=object)

        dijresults = []
        
        count=0
        
        for stage, clusters in enumerate(track):
            temp = []
            for clusterid, leiden in enumerate(clusters):
                temp=self.getDistance(hiddenReps[flag],hiddenReps[count])
                count+=1
            dijresults.append(temp)
        impactFactor = perturbated_genes
        selectedtemp = self.getZandZc(None,selectedstage,track[selectedstage][0],impactfactor = impactFactor,CUDA=CUDA)
        count = 0
        fijresults = []
        for stage, clusters in enumerate(track):
            temp = []
            for clusterid, leiden in enumerate(clusters):
                temp = self.getDistance(selectedtemp,hiddenReps[count])
                count+=1
            fijresults.append(temp)

        delta = np.array(fijresults) - np.array(dijresults)[:,np.newaxis]
        gc.collect()
        
        out = []
        for i in range(impactFactor.shape[0]):
            temp = []
            temp.append(perturbated_genes[i])
            for kk in range(len(track)):
                temp.append(track[kk][0])
            for kk in range(len(track)):
                temp.append(delta[kk][i])
            out.append(temp)

        return out
    def prepare_speed_perturbation_data(self, adata, stage, leiden,raw=None, impactfactors=None):
        if raw is None:
            stageadata = adata[stage]
            clusterAdata = self.stage_cluster[str(stage)][str(leiden)]#stageadata[self.stage_cluster_index[str(stage)][str(leiden)]]#.index.tolist()
            # clusterAdata = stageadata[clusterAdataID]
            adj = clusterAdata.obsp['gcn_connectivities']
            # matrix1 = cupyx.scipy.sparse.csr_matrix(clusterAdata.obsp['gcn_connectivities'])#cp.asarray(clusterAdata.obsp['gcn_connectivities'])
            # matrix2 = cp.asarray(clusterAdata.X)
            # input = matrix1 @ matrix2
            # input = cp.asnumpy(input)
            # input = clusterAdata.obsp['gcn_connectivities'] @ clusterAdata.X
            # if issparse(input):
            #     input = input.toarray()
            #expand [10,3] to [1,10,3]
            input = clusterAdata.X
            data = np.expand_dims(input,axis=0)

            
        else:
            data,adj = raw
            data = np.array(data)
        if impactfactors is not None:
            # print(np.mean(data + data * impactfactors- data))
            data = data.repeat(len(impactfactors),axis=1)

            data =data + data * impactfactors[:, :, np.newaxis, np.newaxis]
        
        if raw is None:
            return [data,adj]#.reshape(-1)
        else:
            return [data,adj]
    def getZ_speedup(self, input,CUDA=False):
        zs = []
        zmeans = []
        zstds = []
        input_adata = input[0]
        input_adj = input[1]
        input_adata = np.array(input_adata)
        input_adata = input_adata.reshpae(input_adata[0]*input_adata[1],input_adata[2],-1)
        input_adj = np.array(input_adj)
        input_adj = input_adj.reshape(input_adj[0]*input_adj[1],input_adj[2],-1)
        loadModelDict = self.model_name#'./'+self.target_directory+'/model_save/'+self.model_name+'.pth'
        vae = VAE(input_adata.shape[2], 256,1024, 64,beta=1,distribution=self.dist)
        if CUDA:
            
            vae.load_state_dict(torch.load(loadModelDict,map_location='cuda:0'))
            # vae = torch.load(loadModelDict)
            vae.to('cuda:0')
        else:
            vae.load_state_dict(torch.load(loadModelDict), map_location=torch.device('cpu'))
            vae.to('cpu')
        
        vae.eval()
        cell_loader = DataLoader(input_adata.astype('float32'), batch_size=1, shuffle=False, num_workers=0)
        
       
        for adj_idx, x in enumerate(cell_loader):
            # if on GPU put mini-batch into CUDA memory
            adj = input_adj[adj_idx]
            if CUDA:
                x = x.to('cuda:0')
                adj = adj.to('cuda:0')
            z = vae.get_latent_representation(x,adj)
            zs+=z.cpu().detach().numpy().tolist()
        zs = np.array(zs)
        return zs
    def perturbation__auto_centroid_speed(self,adata, lastClusters, perturbated_genes,CUDA=False):
        '''
        remove non top genes and tf. compared
        '''

        hiddenReps=[]
        repNodes = []
        flag = -1 #to indicate which cluster in the track to be changed like [40,1,3,2,4,5] 3 is no.2
        clusterids = []
        zc = []
        input_data = []
        input_pertubred = []
        # adatacollection = adata#[]
        
        for i, each in enumerate(lastClusters):
            track = self.getTrack(len(self.stageadata)-1,each)
            for stage, clusters in enumerate(track):
                for clusterid, leiden in enumerate(clusters):
                    input_data.append(self.prepare_speed_perturbation_data(adata, stage, leiden))
        
        # input_data = np.array(input_data)
        input_pertubred_forward = self.prepare_speed_perturbation_data(adata, stage, leiden, raw =input_data, impactfactors = perturbated_genes[0])
        input_pertubred_backward = self.prepare_speed_perturbation_data(adata, stage, leiden, raw =input_data, impactfactors = perturbated_genes[1])
        input_pertubred = np.append(input_pertubred_forward,input_pertubred_backward,axis=0)
        # input_pertubred = np.array(input_pertubred)

        Z_input = self.getZ_speedup(input_data,CUDA).reshape(-1,4,64)
        Z_perturbed = self.getZ_speedup(input_pertubred,CUDA).reshape(-1,4,64)
        input_distance = []
        for i, each in enumerate(Z_input):
            for  j, each1 in enumerate(each):
                for k, each2 in enumerate(each):
                    each1 = each1.reshape(1, -1)
                    each2 = each2.reshape(1, -1)
                    input_distance.append(self.getDistance(each1,each2))
        input_distance = np.array(input_distance)
        input_distance = input_distance.reshape(-1,4,4)
        distance = []
        # print(input_distance.shape)
        for i, each in enumerate(Z_perturbed):
            for j, each1 in enumerate(Z_perturbed[i]):
                if i < len(Z_perturbed)//2:
                    count = i
                else:
                    count = i - len(Z_perturbed)//2
                for k, each2 in enumerate(Z_input[count]):
                    each1 = each1.reshape(1, -1)
                    each2 = each2.reshape(1, -1)
                    distance.append(self.getDistance(each1,each2))
        
        distance = np.array(distance)
        distance = distance.reshape(-1,4,4)
        # print(distance.shape)
        
        #delta1 is first half of delta, delta2 is second half of delta
        delta1 = distance[:len(distance)//2]- input_distance
        delta2 = distance[len(distance)//2:]- input_distance
        # print(delta1.shape)
        # print(delta2.shape)
        return delta1, delta2
        
        # out = []
        # for i in range(impactFactor.shape[0]):
        #     temp = []
        #     temp.append(perturbated_genes[i])
        #     for kk in range(len(track)):
        #         temp.append(track[kk][0])
        #     for kk in range(len(track)):
        #         temp.append(delta[kk][i])
        #     out.append(temp)

        return out
    def get_drug_genes(self,bound):
        drug_target = self.adata.uns['data_drug_overlap_genes']#this should be an attribute of adata object later on
        #drug_target = dict(np.load(drug_cell_type_target,allow_pickle=True).tolist()) #this should be an attribute of adata object later on
        drug_names = list(drug_target.keys())

        drug_target_genes = list(drug_target.values())
        temp_drug_target_genes = drug_target_genes.copy()
        perturbed_genes = []
        
        for temp in temp_drug_target_genes:
            
            out_temp = []
            for each in temp:
                each = each.split(':')
                if each[1] == '+':
                    each = each[0]+':'+str(bound)
                elif each[1] == '-':
                    each = each[0]+':'+str(1/bound)
                out_temp.append(each)
            perturbed_genes.append(out_temp)
        return drug_names, perturbed_genes
    def startAutoPerturbation_online(self,lastCluster,perturbed_genes,CUDA=False):
        '''
        Start the perturbation analysis (online version).

        parameters
        -------------------
        lastCluster: int
            The cluster id of the last cluster in the track
        perturbed_genes: list
            A list of perturbed genes
        CUDA: bool
            Whether to use GPU

        return
        -------------------
        out: dict
            A dictionary of perturbation results
        '''
        out = {}
        out['online'] = {}
        track = self.getTrack(len(self.stageadata)-1,lastCluster)

        perturbed_genes = [perturbed_genes]
        perturbated_gene_ids = []
        impactFactor = []
        for perturbated_gene in perturbed_genes:
            
            temp_perturbated_gene = perturbated_gene.copy()
            temp_bound = []
            perturbated_gene = []
            for each in temp_perturbated_gene:
                each = each.split(':')
                if len(each) > 1:
                    temp_bound.append(float(each[1]))
                    perturbated_gene.append(each[0])
                else:
                    perturbated_gene = temp_perturbated_gene
                    break

            perturbated_gene_id = self.matchSingleClusterGeneDict(self.stageadata[-1],perturbated_gene)
            perturbated_gene_ids.append(perturbated_gene_id)
            temp = np.zeros(shape=(len(self.stageadata[-1].var.index.tolist())))

            for id_each, each in enumerate(perturbated_gene_id):
                temp[each] = temp_bound[id_each]-1
            
            impactFactor.append(temp)
        impactFactor = np.array(impactFactor)

  

        perturbation_results = [[] for i in range(len(track))]
   
        for i, selectedcluster in enumerate(track):
            # print(i)
            threads = []
            self.stageadata[i].obs['leiden'] = self.stageadata[i].obs['leiden'].astype('string')
            perturbation_results[i] += self.perturbation__auto_centroid(self.stageadata[i], self.stageadata, i, selectedcluster[0], track, None,impactFactor,CUDA)
                                      
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            
            for od, each in enumerate(perturbation_results[i]):
                tempout = []
                for kk in range(self.total_stage):
                    tempout.append(each[len(each)-self.total_stage+kk])
                out['online'][str(i)] = tempout

        return out

    def startAutoPerturbation_online_speed(self,lastClusters,perturbed_genes,track_names,CUDA=False):
        '''
        Start the perturbation analysis (online version).
        
        parameters
        -------------------
        lastClusters: list
            A list of last clusters in the track
        perturbed_genes: list
            A list of perturbed genes
        track_names: list
            A list of track names
        CUDA: bool
            Whether to use GPU

        return
        -------------------
        out1: dict
            A dictionary of perturbation results
        out2: dict
            A dictionary of perturbation results

        '''
        out = {}
        out['online'] = {}
        
        temp_track = self.getTrack(len(self.stageadata)-1,lastClusters[0])
        impactFactors = []
        for each_perturbed_genes in perturbed_genes:
            perturbed_genes_temp = [each_perturbed_genes]
            perturbated_gene_ids = []
            impactFactor = []
            for perturbated_gene in perturbed_genes_temp:
                
                temp_perturbated_gene = perturbated_gene.copy()
                temp_bound = []
                perturbated_gene = []
                for each in temp_perturbated_gene:
                    each = each.split(':')
                    if len(each) > 1:
                        temp_bound.append(float(each[1]))
                        perturbated_gene.append(each[0])
                    else:
                        perturbated_gene = temp_perturbated_gene
                        break

                perturbated_gene_id = self.matchSingleClusterGeneDict(self.stageadata[-1],perturbated_gene)
                perturbated_gene_ids.append(perturbated_gene_id)
                temp = np.zeros(shape=(len(self.stageadata[-1].var.index.tolist())))

                for id_each, each in enumerate(perturbated_gene_id):
                    temp[each] = temp_bound[id_each]-1
                
                impactFactor.append(temp)
            impactFactor = np.array(impactFactor)
            impactFactors.append(impactFactor)
  

        perturbation_results = [[] for i in range(len(temp_track))]
        outs = self.perturbation__auto_centroid_speed(self.stageadata,lastClusters,impactFactors,CUDA)
        out1 = {}
        out2 = {}
        for i, each in enumerate(lastClusters):
            track = self.getTrack(len(self.stageadata)-1,each)

            out1[track_names[i]] = {}
            out2[track_names[i]] = {}
            out1[track_names[i]]['online'] = {}
            out2[track_names[i]]['online'] = {}
            for j, selectedcluster in enumerate(track):
                out1[track_names[i]]['online'][str(j)] = outs[0][i][j]
                out2[track_names[i]]['online'][str(j)] = outs[1][i][j]
        
        return out1, out2
    
    def assign_random_direction_to_random_genes(self,random_genes):
        '''
        Build the the sets of random genes with random direction.

        parameters
        -------------------
        random_genes: list
            A of list of random genes

        return
        -------------------
        out: list
            A list of random genes with random direction
        reversed_out: list
            A list of random genes with reversed direction
        '''
        out = []
        reversed_out = []
        temp_random_genes = random_genes.copy()
        for temp in temp_random_genes:
            out_temp = []
            copyout_temp = []
            for each in temp:
                copyeach = each
                flag = np.random.choice(['-','+'])
                # bound = np.random.uniform(1.00001,3)
                bound = 3
                if flag == '+':
                    
                    each = each+':'+str(bound)
                    copyeach = copyeach+':'+str(1/bound)
                elif flag == '-':
                    each = each+':'+str(1/bound)
                    copyeach = copyeach+':'+str(bound)
                out_temp.append(each)
                copyout_temp.append(copyeach)
            out.append(out_temp)
            reversed_out.append(copyout_temp)
        return out,reversed_out
   #~~~~~         
    def startAutoPerturbation(self,lastCluster,bound,mode,CUDA = True,random_genes= None, random_times = None,written=True):
        '''
        Start the perturbation analysis.

        parameters
        -------------------
        lastCluster: int
            The cluster id of the last cluster in the track
        bound: float    
            The perturbation bound
        mode: str
            The perturbation mode, can be 'drug', 'pathway', 'perfect', 'random_background', 'online_random_background'
        CUDA: bool
            Whether to use GPU
        random_genes: list
            A list of random genes
        random_times: int
            The number of random genes
        written: bool
            Whether to write the results to disk

        return
        -------------------
        None
        '''
        track = self.getTrack(len(self.stageadata)-1,lastCluster)

        track_name = str(track[0][0]) 
        for i in range(1,len(track)):
            track_name += '-' + str(track[i][0])
        outs = [[] for i in range(len(track))]
        if mode == 'drug':
            perturbed_items, perturbed_genes = self.get_drug_genes(bound)
        elif mode == 'pathway':
            pathway_gene = self.adata.uns['data_pathway_overlap_genes']
            #pathway_gene = dict(np.load('./'+self.target_directory+'/data_pathway_overlap_genes.npy',allow_pickle=True).tolist())#this should be an attribute of adata object later on
            perturbed_items = list(pathway_gene.keys())
            temp_perturbed_genes = list(pathway_gene.values())
            perturbed_genes = []
            for each in temp_perturbed_genes:
                if type(each)!= list:
                    each = each.tolist()

                perturbed_genes.append(each)
        
        ######
        elif mode == 'perfect':
            perturbed_items, perturbed_genes = self.get_drug_genes(bound)

        elif mode == 'random_background':
        
            perturbated_genes = []
            genelen = len(self.stageadata[0].var.index.tolist())
            genenames = np.array(list(self.stageadata[0].var.index.values))
            shuffled_gene_id = [j for j in range(genelen)]
        elif mode == 'online_random_background':
            bound = 'A'
            genelen = len(self.stageadata[0].var.index.tolist())
            genenames = np.array(list(self.stageadata[0].var.index.values))
            shuffled_gene_id = [j for j in range(genelen)]
            perturbed_genes = []
            perturbed_items = []
            
            for each in range(random_times):
                random_genes = random.randint(1,3)
                perturbed_items.append(str(each))
                random.shuffle(shuffled_gene_id)
                shuffled_gene = genenames[shuffled_gene_id[:random_genes]]
                perturbed_genes.append(shuffled_gene.tolist())
            perturbed_genes,reversed_perturbed_genes = self.assign_random_direction_to_random_genes(perturbed_genes)
        if mode != 'random_background':
            impactFactor = []
            perturbated_gene_ids = []
            for perturbated_gene in perturbed_genes:

                if type(perturbated_gene) != list:
                    perturbated_gene = [perturbated_gene]
                temp_perturbated_gene = perturbated_gene.copy()
                temp_bound = []
                perturbated_gene = []
                for each in temp_perturbated_gene:
                    each = each.split(':')
                    if len(each) > 1:
                        temp_bound.append(float(each[1]))
                        perturbated_gene.append(each[0])
                    else:
                        perturbated_gene = temp_perturbated_gene
                        break

                perturbated_gene_id = self.matchSingleClusterGeneDict(self.stageadata[-1],perturbated_gene)
                perturbated_gene_ids.append(perturbated_gene_id)
                temp = np.zeros(shape=(len(self.stageadata[-1].var.index.tolist())))
                if len(temp_bound) == 0:
                    temp[perturbated_gene_id] = 1
                    temp = temp*(bound-1)
                else:
                    for id_each, each in enumerate(perturbated_gene_id):
                        temp[each] = temp_bound[id_each]-1

                impactFactor.append(temp)
            impactFactor = np.array(impactFactor)


        for i, selectedcluster in enumerate(track):
    
            if '%s_perturbation_deltaD'%mode not in self.adata.uns.keys():
                self.adata.uns['%s_perturbation_deltaD'%mode] = {}
            threads = []
            if mode == 'random_background':
                perturbed_genes = []
                perturbed_items = []
                for each in range(random_times):
                    perturbed_items.append(str(each))
                    random.shuffle(shuffled_gene_id)
                    shuffled_gene = genenames[shuffled_gene_id[:random_genes]]
                    perturbed_genes.append(shuffled_gene.tolist())
                impactFactor = []
                perturbated_gene_ids = []
                for perturbated_gene in perturbed_genes:
                
                    if type(perturbated_gene) != list:
                        perturbated_gene = [perturbated_gene]
                    temp_perturbated_gene = perturbated_gene.copy()
                    temp_bound = []
                    perturbated_gene = []
                    for each in temp_perturbated_gene:
                        each = each.split(':')
                        if len(each) > 1:
                            temp_bound.append(float(each[1]))
                            perturbated_gene.append(each[0])
                        else:
                            perturbated_gene = temp_perturbated_gene
                            break

                    perturbated_gene_id = self.matchSingleClusterGeneDict(self.stageadata[-1],perturbated_gene)
                    perturbated_gene_ids.append(perturbated_gene_id)
                    temp = np.zeros(shape=(len(self.stageadata[-1].var.index.tolist())))
                    if len(temp_bound) == 0:
                        temp[perturbated_gene_id] = 1
                        temp = temp*(bound-1)
                    else:
                        for id_each, each in enumerate(perturbated_gene_id):
                            temp[each] = temp_bound[id_each]-1
                    
                    impactFactor.append(temp)
                impactFactor = np.array(impactFactor)

            self.stageadata[i].obs['leiden'] = self.stageadata[i].obs['leiden'].astype('string')
            # outs[i] = self.perturbationthread(self,i, selectedcluster[0], track, bound,impactFactor,CUDA)
            if mode == 'perfect':
                outs[i] +=self.perfect_perturbation__auto_centroid(self.stageadata[i], self.stageadata, i, selectedcluster[0], track, bound,impactFactor,CUDA)

            else:
                outs[i] += self.perturbation__auto_centroid(self.stageadata[i], self.stageadata, i, selectedcluster[0], track, bound,impactFactor,CUDA)

            for od, each in enumerate(outs[i]):
                if written == True:
                    # with open('./'+self.target_directory+'/tsdg2_%s_%s.csv'%(mode,bound),"a+") as f:
                        # f.write('%s,%d-%d-%d-%d,%.7f,%.7f,%.7f,%.7f\n'%(perturbed_items[od],each[1], each[2], each[3], each[4],each[5],each[6],each[7],each[8]))
                    pass #fix this later on
                else:
                    
                    # for  name in perturbed_items:
                    if str(bound) not in self.adata.uns['%s_perturbation_deltaD'%mode].keys():
                        self.adata.uns['%s_perturbation_deltaD'%mode][str(bound)] = {}
                    if track_name not in self.adata.uns['%s_perturbation_deltaD'%mode][str(bound)].keys():
                        self.adata.uns['%s_perturbation_deltaD'%mode][str(bound)][track_name] = {}
                    if perturbed_items[od] not in self.adata.uns['%s_perturbation_deltaD'%mode][str(bound)][track_name].keys():
                        self.adata.uns['%s_perturbation_deltaD'%mode][str(bound)][track_name][perturbed_items[od]] = {}
                    if str(i) not in self.adata.uns['%s_perturbation_deltaD'%mode][str(bound)][track_name][perturbed_items[od]].keys():
                        tempout = []
                        for kk in range(self.total_stage):
                            tempout.append(each[len(each)-self.total_stage+kk])
                        # if self.mode == 'random_background':
                        #     if bound < 1/bound:
                        #         bound = 1/bound
                        #     self.adata.uns['%s_perturbation_deltaD'%mode][str(bound)][track_name][perturbed_items[od]+str(bound)][str(i)] = tempout
                        # else:
                        self.adata.uns['%s_perturbation_deltaD'%mode][str(bound)][track_name][perturbed_items[od]][str(i)] = tempout
        if mode == 'online_random_background':
            bound = 'B'
            self.hiddenReps = []
            self.perturb_stage_data_mean = []
            reversed_perturbed_genes
            impactFactor = []
            perturbated_gene_ids = []
            for perturbated_gene in reversed_perturbed_genes:
            
                if type(perturbated_gene) != list:
                    perturbated_gene = [perturbated_gene]
                temp_perturbated_gene = perturbated_gene.copy()
                temp_bound = []
                perturbated_gene = []
                for each in temp_perturbated_gene:
                    each = each.split(':')
                    if len(each) > 1:
                        temp_bound.append(float(each[1]))
                        perturbated_gene.append(each[0])
                    else:
                        perturbated_gene = temp_perturbated_gene
                        break

                perturbated_gene_id = self.matchSingleClusterGeneDict(self.stageadata[-1],perturbated_gene)
                perturbated_gene_ids.append(perturbated_gene_id)
                temp = np.zeros(shape=(len(self.stageadata[-1].var.index.tolist())))
                if len(temp_bound) == 0:
                    temp[perturbated_gene_id] = 1
                    temp = temp*(bound-1)
                else:
                    for id_each, each in enumerate(perturbated_gene_id):
                        temp[each] = temp_bound[id_each]-1
                
                impactFactor.append(temp)
            impactFactor = np.array(impactFactor)
        
            outs = [[] for i in range(len(track))]
            for i, selectedcluster in enumerate(track):
                threads = []
                outs[i] += self.perturbation__auto_centroid(self.stageadata[i], self.stageadata, i, selectedcluster[0], track, bound,impactFactor,CUDA)
                # threads.append(self.perturbationthread(self,outs, i, selectedcluster[0], track, bound,impactFactor,CUDA))

                for od, each in enumerate(outs[i]):
                    if str(bound) not in self.adata.uns['%s_perturbation_deltaD'%mode].keys():
                        self.adata.uns['%s_perturbation_deltaD'%mode][str(bound)] = {}
                    if track_name not in self.adata.uns['%s_perturbation_deltaD'%mode][str(bound)].keys():
                        self.adata.uns['%s_perturbation_deltaD'%mode][str(bound)][track_name] = {}
                    if perturbed_items[od] not in self.adata.uns['%s_perturbation_deltaD'%mode][str(bound)][track_name].keys():
                        self.adata.uns['%s_perturbation_deltaD'%mode][str(bound)][track_name][perturbed_items[od]] = {}
                    if str(i) not in self.adata.uns['%s_perturbation_deltaD'%mode][str(bound)][track_name][perturbed_items[od]].keys():
                        tempout = []
                        for kk in range(self.total_stage):
                            tempout.append(each[len(each)-self.total_stage+kk])
                        self.adata.uns['%s_perturbation_deltaD'%mode][str(bound)][track_name][perturbed_items[od]][str(i)] = tempout
    def run(self,mode,log2fc,inplace=False,random_times = 100,random_genes = 2,CUDA = True):
        '''
        Perform perturbation.

        parameters
        -------------------
        mode: str
            perturbation mode, 'drug', 'pathway', 'random_background', 'online_random_background', 'perfect'
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

        return
        -------------------
        None
        '''
        if inplace == False:
            written=True
        else:
            written=False
        if mode == 'drug':
            for i in self.tracks.keys():

                print(i)
                self.startAutoPerturbation(i,log2fc,mode,written =written,CUDA=CUDA)
                self.startAutoPerturbation(i,1/log2fc,mode,written =written,CUDA=CUDA)
                self.hiddenReps = []
                self.perturb_stage_data_mean = []
        elif mode == 'pathway':
            for i in self.tracks.keys():
                self.startAutoPerturbation(i,log2fc,mode,written =written,CUDA=CUDA)
                self.startAutoPerturbation(i,1/log2fc,mode,written =written,CUDA=CUDA)
                self.hiddenReps = []
                self.perturb_stage_data_mean = []
        elif mode == 'random_background':
            for i in self.tracks.keys():
                self.startAutoPerturbation(i,log2fc,mode,written = written,random_times=random_times,random_genes=random_genes,CUDA=CUDA)
                self.startAutoPerturbation(i,1/log2fc,mode, written = written,random_times=random_times,random_genes=random_genes,CUDA=CUDA)
                self.hiddenReps = []
                self.perturb_stage_data_mean = []
        elif mode == 'online_random_background':
            for i in self.tracks.keys():
                self.startAutoPerturbation(i,1,mode,random_times=random_times,written =written,CUDA=CUDA)
        elif mode == 'perfect':
            for i in self.tracks.keys():

                print(i)
                self.startAutoPerturbation(i,log2fc,mode,written =written,CUDA=CUDA)
                self.startAutoPerturbation(i,1/log2fc,mode,written =written,CUDA=CUDA)
                self.hiddenReps = []
                self.perturb_stage_data_mean = []
    def run_online_speed(self, allTracks:bool,perturbated_gene,perturbated_gene_reversed, unit_name,stage = None, lastCluster=None,CUDA=False):
        '''
        Perform online perturbation.
        
        parameters
        -------------------
        allTracks: bool
            Using one track or all tracks

        perturbated_gene: dict
            gene to be perturbed and the regulated intensity ({a:0.5, b: 2.5, c:0.5...})
        perturbated_gene_reversed: dict
            gene to be perturbed and the regulated intensity ({a:2.0, b: 0.4, c:2.0...} (reversed log2fc to the original)

        unit_name: str
            name of the unit to be perturbed
        stage: 
            stage to be perturbed, if None choose all
        CUDA: bool
            whether to use CUDA

        return
        -------------------
        perturbation_score: float
            perturbation score
        pval: float
            p value
        out_deltaD: dict
            deltaD of the perturbed unit
        '''
        import time

        online_analyst = perturbationAnalysis(self.adata,self.idrem_dir,stage=stage,allTracks=allTracks)
        # else:

            # online_analyst = perturbationAnalysis(self.adata,os.path.join(self.target_directory,'idremVizCluster'),allTracks=allTracks)
        perturbated_gene = perturbated_gene.split(',')
        perturbated_gene_reversed = perturbated_gene_reversed.split(',')
        if allTracks != True:
            out1 = {}
            out2 = {}
            track = self.getTrack(len(self.stageadata)-1,lastCluster)
            track_name = str(track[0][0])
            for i in range(1,len(track)):
                track_name += '-' + str(track[i][0])
            
            out1[track_name] = self.startAutoPerturbation_online(lastCluster,perturbated_gene,CUDA=CUDA)
            out2[track_name] = self.startAutoPerturbation_online(lastCluster,perturbated_gene_reversed,CUDA=CUDA)
            track = self.getTrack(len(self.stageadata)-1,lastCluster)

            perturbation_score, pval,out_deltaD = online_analyst.online_analysis([track_name,[out1,out2]])
            self.hiddenReps = []
            self.perturb_stage_data_mean = []
        else:
            out1 = {}
            out2 = {}
            last_clusters = []
            track_names = []
            for i in self.tracks.keys():
            
                track = self.getTrack(len(self.stageadata)-1,i)
                last_clusters.append(track[-1][0])
                track_name = str(track[0][0]) 
                for i in range(1,len(track)):
                    track_name += '-' + str(track[i][0])
                track_names.append(track_name)
            
            out1,out2 = self.startAutoPerturbation_online_speed(last_clusters,[perturbated_gene,perturbated_gene_reversed],track_names,CUDA=CUDA)
            # out2 = self.startAutoPerturbation_online_speed(last_clusters,perturbated_gene_reversed,track_names,CUDA=CUDA)
               
            step6_start = time.time()
            perturbation_score, pval,out_deltaD = online_analyst.online_analysis([out1,out2])
            step6_end = time.time()
            print('step6 time: ', step6_end - step6_start)
        return perturbation_score, pval,out_deltaD
    def run_online(self, allTracks:bool,perturbated_gene,perturbated_gene_reversed, unit_name,stage = None, lastCluster=None,CUDA=False):
        '''
        Perform online perturbation.

        parameters
        -------------------
        allTracks: bool
            One track or all tracks
        stage: int
            stage to be perturbed
        lastCluster: int
            last cluster to be perturbed (if allTracks is False)
        perturbated_gene: list
            gene to be perturbed format a:0.5, b: 2.5, c:0.5...
        perturbated_gene_reversed: list
            gene to be perturbed format a:2.0, b: 0.4, c:2.0... (reversed log2fc to the original)
        unit_name: str
            name of the unit to be perturbed
        stage: int
            stage to be perturbed, if None choose all
        CUDA: bool
            whether to use CUDA

        return
        -------------------
        perturbation_score: np.float
            perturbation score
        pval: np.float
            p value
        '''
        import time

        online_analyst = perturbationAnalysis(self.adata,self.idrem_dir,stage=stage,allTracks=allTracks)
        # else:

            # online_analyst = perturbationAnalysis(self.adata,os.path.join(self.target_directory,'idremVizCluster'),allTracks=allTracks)
        perturbated_gene = perturbated_gene.split(',')
        perturbated_gene_reversed = perturbated_gene_reversed.split(',')
        if allTracks != True:
            out1 = {}
            out2 = {}
            track = self.getTrack(len(self.stageadata)-1,lastCluster)
            track_name = str(track[0][0])
            for i in range(1,len(track)):
                track_name += '-' + str(track[i][0])
            
            out1[track_name] = self.startAutoPerturbation_online(lastCluster,perturbated_gene,CUDA=CUDA)
            out2[track_name] = self.startAutoPerturbation_online(lastCluster,perturbated_gene_reversed,CUDA=CUDA)
            track = self.getTrack(len(self.stageadata)-1,lastCluster)

            perturbation_score, pval,out_deltaD = online_analyst.online_analysis([track_name,[out1,out2]])
            self.hiddenReps = []
            self.perturb_stage_data_mean = []
        else:
            out1 = {}
            out2 = {}
            for i in self.tracks.keys():
            
                track = self.getTrack(len(self.stageadata)-1,i)
                track_name = str(track[0][0])
                for j in range(1,len(track)):
                    track_name += '-' + str(track[j][0])
         
                out1= self.startAutoPerturbation_online(track[-1][0],perturbated_gene,CUDA=CUDA)
                out2 = self.startAutoPerturbation_online(track[-1][0],perturbated_gene_reversed,CUDA=CUDA)
                self.hiddenReps = []
                self.perturb_stage_data_mean = []
            step6_start = time.time()
            perturbation_score, pval,out_deltaD = online_analyst.online_analysis([out1,out2])
            step6_end = time.time()
            print('step6 time: ', step6_end - step6_start)
        return perturbation_score, pval,out_deltaD
    def analysis(self,mode,log2fc,all=True,stage=None):
        '''
        Analysis of perturbation results
        
        parameters
        ----------------
        mode: str
            The mode is choosing from ['drug', 'pathway', 'online']
        log2fc: float
            log2fc is the log2 fold change of perturbation
        all: bool
            all is whether to analysis all tracks or one track
        stage: int
            stage is the stage to be analysis, if all is True, stage is None
        '''

        self.adata.obs['leiden'] = self.adata.obs['leiden'].astype('string')
        self.adata.obs['stage'] = self.adata.obs['stage'].astype('string')
        analyst = perturbationAnalysis(self.adata,self.idrem_dir, stage=stage,log2fc = log2fc, mode = mode)
       
        
        temp = analyst.main_analysis(track_to_analysis = 'all', all=all, score='avg_backScore', items=None)#read item from disk for now
        if '%s_perturbation_score'%mode not in self.adata.uns.keys():
            self.adata.uns['%s_perturbation_score'%mode] = {}
        
        self.adata.uns['%s_perturbation_score'%mode][str(log2fc)] = temp
        self.adata.uns['%s_perturbation_score'%mode][str(1/log2fc)] = temp
        