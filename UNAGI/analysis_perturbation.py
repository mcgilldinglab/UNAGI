#drug analysis
#load results csv which are distants.
import csv
import json
import os
import scanpy as sc
import numpy as np
import pandas as pd
from scipy.stats import norm
class perturbationAnalysis:
    def __init__(self,adata,target_directory,stage = None, log2fc=None,mode=None,allTracks=None):
        self.adata = adata
        self.log2fc = log2fc
        self.mode = mode
        self.target_directory = target_directory
        self.stage = stage
        self.allTracks = allTracks
        self.idrem_path = self.target_directory#os.path.join(self.target_directory,'idremVizCluster/')#'./'+ self.target_directory +'/idremVizCluster/'
    def readIdremJson(self, filename):
        # print('getting Target genes from ', filename)
        path = os.path.join(self.idrem_path,filename ,'DREM.json')
        f=open(path,"r")
        lf=f.readlines()
        f.close()
        lf="".join(lf)
        lf=lf[5:-2]+']'
        tt=json.loads(lf,strict=False)
        return tt
    def getTendency(self,filename):
        '''
        get the tendency of each path
        args:
        path: the file path of IDREM results
        
        return:
        out: a list of tendency of each path
        '''
        out = []
        tt = self.readIdremJson(filename)
        temp = np.array(tt[8])
        idrem_genes = temp[1:,0].tolist()
        tendency = (temp[1:,-1].astype(float)- temp[1:,1].astype(float))/3
        # stage_expression = temp[1:,1:5].astype(float)
        return tendency

    def getTendencyFromIDREM(self):
        '''
        get the tendency of each path
        args:
        path: the file path of IDREM results
        
        return:
        out: a list of tendency of each path
        '''
        out = {}
        filenames = os.listdir(self.idrem_path)
        for each in filenames:
            name = each
            if each[0] != '.':
                each = each.split('.')[0].split('-')[-1].split('n')
                for each1 in each:
                    out[each1] = self.getTendency(name)
                
        return out
    def get_tracks(self):
        # dirctory = dirctory+'_0.5.csv'
        # k = open(dirctory)#open('./iterativeTrainingNOV26/4/perturbation0.5ImportantTrack.csv')
        # kk = csv.reader(k,delimiter=',')
        if 'online_random_background_perturbation_deltaD' in list(self.adata.uns.keys()):
            tracks = self.adata.uns['online_random_background_perturbation_deltaD']['A'].keys()
        else:
            tracks = self.adata.uns['%s_perturbation_deltaD'%self.mode][str(self.log2fc)].keys()
        # tracks = []
        # for each in list(deltaD.keys()):
        #     tracks.append(each)

        return list(set(tracks))
    def load_online(self,deltaD=None,sanity=False,track_percentage= None):
        out = {}
        if sanity == True:
            k1 = self.adata.uns['online_random_background_perturbation_deltaD']['A'] #track name, times, i
            k2 = self.adata.uns['online_random_background_perturbation_deltaD']['B']
        else:
            k1 = deltaD[0]#{trackname, 'online', 'i'}
            k2 = deltaD[1]
        pathwaydic1= {}

        for each in list(k1.keys()):

            if each not in pathwaydic1.keys():
                
                for each1 in list(k1[each].keys()):
                    if each1 not in pathwaydic1.keys():
                        pathwaydic1[each1] = {}
                    
                    if each not in pathwaydic1[each1].keys():
                        pathwaydic1[each1][each] = []
                    total_len = len(k1[each][each1])
                    if track_percentage is not None:
                        pathwaydic1[each1][each] = [np.array(k1[each][each1][str(i)]) for i in range(total_len)]*np.array(track_percentage[each])
                    else:
                        pathwaydic1[each1][each] = [np.array(k1[each][each1][str(i)]) for i in range(total_len)]
        pathwaydic2= {}
        for each in list(k2.keys()):

            if each not in pathwaydic2.keys():
                
                for each1 in list(k2[each].keys()):
                    if each1 not in pathwaydic2.keys():
                        pathwaydic2[each1] = {}
                    
                    if each not in pathwaydic2[each1].keys():
                        pathwaydic2[each1][each] = []
                    total_len = len(k2[each][each1])
                    if track_percentage is not None:
                        pathwaydic2[each1][each] = [np.array(k2[each][each1][str(i)]) for i in range(total_len)]*np.array(track_percentage[each]) #each is track, each1 is item name
                    else:
                        pathwaydic2[each1][each] = [np.array(k2[each][each1][str(i)]) for i in range(total_len)]
        if sanity == False:
            for each in list(pathwaydic1.keys()):
                temp1 = []
                temp2 = []
                out_delta = []
                for i in range(len(np.sum(np.array(list(pathwaydic1[each].values())),axis=0))):
                    if self.stage is None:
                        out_delta.append(np.sum(np.array(list(pathwaydic1[each].values())),axis=0)[i])
                        temp1.append(self.calculateScore(np.sum(np.array(list(pathwaydic1[each].values())),axis=0)[i],i))
                        temp2.append(self.calculateScore(np.sum(np.array(list(pathwaydic2[each].values())),axis=0)[i],i))
                    else:
                        if i != self.stage:
                            continue
                        else:
                            self.out_delta = np.sum(np.array(list(pathwaydic1[each].values())),axis=0)[i]
                            temp1.append(self.calculateScore(np.sum(np.array(list(pathwaydic1[each].values())),axis=0)[i],i))
                            temp2.append(self.calculateScore(np.sum(np.array(list(pathwaydic2[each].values())),axis=0)[i],i))
                temp1 = np.array(temp1)
                temp2 = np.array(temp2)
                # out[each] =(np.abs(np.mean(temp1[:,0]))+np.abs(np.mean(temp2[:,0])))/2
                out[each] =np.sqrt(((np.abs(np.mean(temp1[:,0]))+np.abs(np.mean(temp2[:,0])))/2) * ((np.abs(np.mean(temp1[:,1]))+np.abs(np.mean(temp2[:,1])))/2))
                if self.stage is None:
                    self.out_delta = np.mean(out_delta,axis=0)
                # out[each] = (np.abs(np.sum(np.array(list(pathwaydic1[each].values())),axis=0))+np.abs(np.sum(np.array(list(pathwaydic2[each].values())),axis=0)))/2 #total item-delta
        if track_percentage is None and sanity == True:
            out = {}
            
            for each in list(k1.keys()):
                out[each] = []
                for item in list(k1[each].keys()):
                    temp1 = []
                    temp2 = []
                    for i in range(len(np.sum(np.array(list(k1[each][item].values())),axis=0))):
                        if self.stage is None:
                            temp1.append(self.calculateScore(np.array(list(k1[each][item].values()))[i],i))
                            temp2.append(self.calculateScore(np.array(list(k2[each][item].values()))[i],i))
                        else:
                            if i != self.stage:
                                continue
                            else:
                                temp1.append(self.calculateScore(np.array(list(k1[each][item].values()))[i],i))
                                temp2.append(self.calculateScore(np.array(list(k2[each][item].values()))[i],i))
                    temp1 = np.array(temp1)
                    temp2 = np.array(temp2)
                    # out[each].append((np.abs(np.mean(temp1[:,0]))+np.abs(np.mean(temp2[:,0])))/2) 
                    out[each].append(np.sqrt(((np.abs(np.mean(temp1[:,0]))+np.abs(np.mean(temp2[:,0])))/2) * ((np.abs(np.mean(temp1[:,1]))+np.abs(np.mean(temp2[:,1])))/2)))
                    # out[each].append((np.abs(np.array(list(k1[each][item].values())))+np.abs(np.array(list(k2[each][item].values()))))/2)) 
        elif track_percentage is not None and sanity == True:
            out = {}
            for each in list(pathwaydic1.keys()):
                temp1 = []
                temp2 = []
                for i in range(len(np.sum(np.array(list(pathwaydic1[each].values())),axis=0))):
                    if self.stage is None:
                        temp1.append(self.calculateScore(np.sum(np.array(list(pathwaydic1[each].values())),axis=0)[i],i))
                        temp2.append(self.calculateScore(np.sum(np.array(list(pathwaydic2[each].values())),axis=0)[i],i))
                    else:
                        if i != self.stage:
                            continue
                        else:
                            temp1.append(self.calculateScore(np.sum(np.array(list(pathwaydic1[each].values())),axis=0)[i],i))
                            temp2.append(self.calculateScore(np.sum(np.array(list(pathwaydic2[each].values())),axis=0)[i],i))
                temp1 = np.array(temp1)
                temp2 = np.array(temp2)
                # out[each] = (np.abs(np.mean(temp1[:,0]))+np.abs(np.mean(temp2[:,0])))/2
                out[each] =np.sqrt(((np.abs(np.mean(temp1[:,0]))+np.abs(np.mean(temp2[:,0])))/2) * ((np.abs(np.mean(temp1[:,1]))+np.abs(np.mean(temp2[:,1])))/2))
                # out[each] = (np.abs(np.sum(np.array(list(pathwaydic1[each].values())),axis=0))+np.abs(np.sum(np.array(list(pathwaydic2[each].values())),axis=0)))/2
        return out
    def load(self,data_pathway_overlap_genes,track_percentage,all=True,sanity=False):
        if sanity == True:
            k1 = self.adata.uns['random_background_perturbation_deltaD'][str(self.log2fc)]
            k2 = self.adata.uns['random_background_perturbation_deltaD'][str(1/self.log2fc)]
        else:
            k1 = self.adata.uns['%s_perturbation_deltaD'%self.mode][str(self.log2fc)]
            k2 = self.adata.uns['%s_perturbation_deltaD'%self.mode][str(1/self.log2fc)]
        # self.stageadata[i].uns['%s_perturbation_deltaD'%mode][str(bound)][track_name][name]
        pathwaydic1= {}
        for each in list(k1.keys()):

            if each not in pathwaydic1.keys():
                
                for each1 in list(k1[each].keys()):
                    if each1 not in pathwaydic1.keys():
                        pathwaydic1[each1] = {}
                    
                    if each not in pathwaydic1[each1].keys():
                        pathwaydic1[each1][each] = []
                    total_len = len(k1[each][each1])
                    
                    if all == True:
                        pathwaydic1[each1][each] = [np.array(k1[each][each1][str(i)]) for i in range(total_len)]*np.array(track_percentage[each])
                    else:
                        pathwaydic1[each1][each] = [np.array(k1[each][each1][str(i)]) for i in range(total_len)]
       
        pathwaydic2= {}
#each 1 is the item name, each is the track name
        for each in list(k2.keys()):

            if each not in pathwaydic2.keys():
                
                for each1 in list(k2[each].keys()):
                    if each1 not in pathwaydic2.keys():
                        pathwaydic2[each1] = {}
                    
                    if each not in pathwaydic2[each1].keys():
                        pathwaydic2[each1][each] = []
                    total_len = len(k2[each][each1])
                    if all == True:
                        pathwaydic2[each1][each]= [np.array(k2[each][each1][str(i)]) for i in range(total_len)]*np.array(track_percentage[each])
                    else:
                        pathwaydic2[each1][each]= [np.array(k2[each][each1][str(i)]) for i in range(total_len)]
        if all:
            out = {}
            for each in list(pathwaydic1.keys()):
                out[each] = {}
                temp1 = []
                temp2 = []
                temp = []
                for i in range(len(np.sum(np.array(list(pathwaydic1[each].values())),axis=0))):
                    temp1.append(self.calculateScore(np.sum(np.array(list(pathwaydic1[each].values())),axis=0)[i],i))
                    temp2.append(self.calculateScore(np.sum(np.array(list(pathwaydic2[each].values())),axis=0)[i],i))
                temp1 = np.array(temp1)
                temp2 = np.array(temp2)

                temp.append(np.sqrt(((np.abs(np.mean(temp1[:,0]))+np.abs(np.mean(temp2[:,0])))/2) * ((np.abs(np.mean(temp1[:,1]))+np.abs(np.mean(temp2[:,1])))/2)))
                # temp.append((np.abs(np.mean(temp1[:,0]))+np.abs(np.mean(temp2[:,0])))/2) 
                # temp.append((np.mean(temp1[:,0])-np.mean(temp2[:,0]))/2 * ((np.abs(np.mean(temp1[:,1]))+np.abs(np.mean(temp2[:,1])))/2)) 
                out[each]['total'] = np.array(temp)
        else:
            out = {}
            max = 0
            pw = None
            tr = None
            for each in list(pathwaydic1.keys()): #each is the pathway name
                out[each] = {}
                
                for item in list(pathwaydic1[each].keys()):#item is the track name
                    temp1 = []
                    temp2 = []
                    temp = []
                    for i in range(len(np.array(list(pathwaydic1[each][item])))):
                        
                        temp1.append(self.calculateScore(np.array(list(pathwaydic1[each][item]))[i],i))
                        temp2.append(self.calculateScore(np.array(list(pathwaydic2[each][item]))[i],i))
                    temp1_copy = np.array(temp1)
                    temp2_copy = np.array(temp2)

                    # temp.append((np.abs(np.mean(temp1_copy[:,0]))+np.abs(np.mean(temp2_copy[:,0])))/2 )
                    
                    temp.append(np.sqrt(((np.abs(np.mean(temp1_copy[:,0]))+np.abs(np.mean(temp2_copy[:,0])))/2) * ((np.abs(np.mean(temp1_copy[:,1]))+np.abs(np.mean(temp2_copy[:,1])))/2)))
                    # temp.append((np.mean(temp1[:,0])-np.mean(temp2[:,0]))/2* ((np.abs(np.mean(temp1[:,1]))+np.abs(np.mean(temp2[:,1])))/2)) 
                    out[each][item] = np.array(temp)

        if sanity == False:

            for each in list(out.keys()):
                if each not in data_pathway_overlap_genes:
                    del out[each]

        return out
    
#convert distance to scores and some statistics

    def get_cluster_data_size(self,stage,cluster):

        stagedata = self.adata[self.adata.obs['stage'] == str(stage)]#.index.tolist()
        # stagedata = self.adata[stageids]
       
        clusterids = stagedata.obs[stagedata.obs['leiden'] == str(cluster)].index.tolist()
        clusterdata = stagedata[clusterids]
        return len(clusterdata)
    def get_track_percentage(self, tracks):
        percentage = {}
        import time
        stageadatas = [self.adata[self.adata.obs[self.adata.obs['stage'] == stage].index.tolist()] for stage in self.adata.obs['stage'].unique()]
        for each_track in tracks:
           
            if '-' not in each_track:
                continue
            percentage[each_track] = 0
            clusters = each_track.split('-')
        
            for stage, each_cluster in enumerate(clusters):
                
                cluster_size = stageadatas[stage].obs['leiden'].value_counts().to_dict()[each_cluster]#self.get_cluster_data_size(stageadatas[stage],each_cluster)
                percentage[each_track] += cluster_size
            percentage[each_track] /= len(self.adata)


        return percentage
    def calculateScore(self,delta,flag,weight=100):
        '''
        flag: the stage of selected clsuter
        delta: perturbation distance - simulation distance (in z space)
        
        '''
        out = 0
        out1 = 0
        separate = []
        for i, each in enumerate(delta):
            
            if i != flag:
                # print((1-1/(1+np.exp(weight*each*np.sign(i-flag)))-0.5)/0.5)
                out+=(1-1/(1+np.exp(weight*each*np.sign(i-flag)))-0.5)/0.5
                out1+=np.abs((1-1/(1+np.exp(weight*each))-0.5)/0.5)
                # separate.append((1-1/(1+np.exp(weight*each))-0.5)/0.5)
                # separate.append((1-1/(1+np.exp(weight*each*np.sign(i-flag)))-0.5)/0.5)
            # separate[0] = out/3 #just for test
            # out = separate[0]#test
   
        return out/(len(delta)-1), out1/(len(delta)-1)

    def calculateAvg(self,pathwaydic,tracklist='all',sanity=False):
        
        perturbationresultdic = {}
        perturbationresultdic['backScore'] = []
        for each in list(pathwaydic.keys()):

            perturbationresultdic[each] = {}
            

            for track in list(pathwaydic[each].keys()):
                clusterType = track.split('-')
                if tracklist != 'all':
                    target = []
                    for t in tracklist:
                        target.append(str(t))
                    #target = str(track)#['3','16']
                #target = ['3','16','4','28','23','19','9','13']
    #             if clusterType[0] not in target:
    #                 continue
                perturbationresultdic[each][track] = {}
                perturbationresultdic[each][track]['backScore'] = []
                for flag, each_perturbation in enumerate(pathwaydic[each][track]):
                    out = each_perturbation
                    perturbationresultdic[each][track]['backScore'] = out
                    
                
                perturbationresultdic[each][track]['avg_backScore'] = perturbationresultdic[each][track]['backScore']
        return perturbationresultdic



    def getStatistics(self,perturbationresultdic):
        avg_backScore = []
        backScore = []

        tempkey = list(perturbationresultdic.keys())[-1]
        key = list(perturbationresultdic.keys())
        
        # avg_backScore.pop()
        # backScore.pop()
        track_name = list(perturbationresultdic[key[-1]].keys())
        for each in track_name:
            avg_backScore.append([])
            backScore.append([])
        pathway_name_order = []
        key.remove('backScore')
        pathway_name_order = []
        for i in range(len(backScore)):
            for each in key:
                pathway_name_order.append(each)
                backScore[i].append(perturbationresultdic[each][track_name[0]]['backScore'])
                avg_backScore[i].append(perturbationresultdic[each][track_name[0]]['avg_backScore'])
                

        return  avg_backScore,  backScore,track_name,pathway_name_order

    def fitlerOutNarrowPathway(self, scores, sanity_scores, pathway_names, pathway_name_order):
        '''
        for this function, only get pathway cdf in a specific path
        
        args:
        scores: the score of all pathways in a track
        pathway_names: name of the pathway
        pathway_name_order: the pathway name order
        
        perturbationresultdic: perturbation statistic results
        pathdic: scores of pathways in every path
        sanity_pathdic: sanity (random) scores of pathways in every path
        scoreindex: name of score to print
        flag:0: pushback, 1: pushforward
        '''
        
        sanity_scores = np.array(sanity_scores)
        # print('sanity mean: ', sanity_scores.mean())
        # print('sanity std: ', sanity_scores.std())
        top_compounds = []
        down_compounds = []
        record_top = []
        record_down = []

    
        for i,each in enumerate(scores):

            if float(each) >= sanity_scores.mean()+sanity_scores.std():
                top_compounds.append(pathway_names[i])
                record_top.append(i)
            elif  float(each) <= sanity_scores.mean()-sanity_scores.std():
                
                down_compounds.append(pathway_names[i])
                record_down.append(i)
        scores = np.array(scores)
        # print(down_compounds)
        filtered_top_score = scores[record_top]
        filtered_down_score = scores[record_down]
        filtered_top_index = np.argsort(filtered_top_score).tolist()
        filtered_top_index.reverse()
        filtered_down_index = np.argsort(filtered_down_score).tolist()

        filtered_top_score = sorted(filtered_top_score,reverse=True)
        filtered_down_score = sorted(filtered_down_score)
        
        top_compounds = np.array(top_compounds)
        down_compounds = np.array(down_compounds)
        top_compounds = top_compounds[filtered_top_index]
        down_compounds = down_compounds[filtered_down_index]
        
        final_down_compounds = []
        final_top_compounds = []
        for i, each in enumerate(filtered_top_score):
            cdf = norm.cdf(each, sanity_scores.mean(),sanity_scores.std())
            # if (1.000-cdf) * len(filtered_top_score) / (i + 1)  < 0.05:
            if (1.000-cdf) <0.05:
                final_top_compounds.append(top_compounds[i])
        for i, each in enumerate(filtered_down_score):
            cdf = norm.cdf(each, sanity_scores.mean(),sanity_scores.std())
            if cdf * len(filtered_down_score) / (i + 1) < 0.05:
                # print('down compound:',down_compounds[i])
                final_down_compounds.append(down_compounds[i])
        return top_compounds,down_compounds
    def getTopDownPathways(self,pathwaydic,track_percentage,track,all=all, flag=0):
        '''
        flag: 0:both pushback score and pushforward score; 1 only pushback score, -1 only pushforward score
        
        '''

        perturbationresultdic = self.calculateAvg(pathwaydic,tracklist=track)
        sanity_pathwaydic = self.load( None ,track_percentage,all=all,sanity=True)
        sanity_perturbationresultdic = self.calculateAvg(sanity_pathwaydic,sanity=True,tracklist=track)
        

        pathways = list(perturbationresultdic.keys())
        pathways.remove('backScore')

        # pathdic,sanity_pathdic = conver_pathway_track_score_to_track_pathway_score(pathways,perturbationresultdic,sanity_perturbationresultdic,scoreindex)
        sanity_avg_backScore, sanity_backScore,sanity_track_name,_ = self.getStatistics(sanity_perturbationresultdic)

        avg_backScore, backScore,track_name,pathway_name_order = self.getStatistics(perturbationresultdic)

        outtext=[]
        results= {}
        for i, each in enumerate(avg_backScore):
            
            results[track_name[i]] = {}
            
            top_compounds,down_compounds=self.fitlerOutNarrowPathway(each,sanity_avg_backScore[i],pathways,pathway_name_order)
            # print(down_compounds)
            results[track_name[i]]['top_compounds']={}
            results[track_name[i]]['down_compounds']= {}
            for j,each in enumerate(top_compounds):
                results[track_name[i]]['top_compounds'][str(j)] = top_compounds[j]
            for j,each in enumerate(down_compounds):  
                results[track_name[i]]['down_compounds'][str(j)] = down_compounds[j]
        results = pd.json_normalize(results)
        results.columns = results.columns.str.split(".").map(tuple)
        results = results.stack([0, 1]).reset_index(0, drop=True)
        results = results.transpose()
        # return results
        results = results.reset_index()

        results['index'] = results['index'].astype('int32')
        results = results.set_index(['index'])
        results =results.sort_index()
    
        return results

    def conver_pathway_track_score_to_track_pathway_score(self,pathway,perturbationresultdic,sanity_perturbationresultdic,score,track):
        '''
        previously the dict is pathway-track-scores, convert it to track-scores
        
        args:
        pathway: a list of names of pathways
        perturbationresultdic: perturbation statistic results
        '''
        
        pathdic = {}
        sanity_pathdic = {}
        for path in list(perturbationresultdic[pathway[0]].keys()):

            if path == 'statistic':
                continue
            if track == 'all':
                pathdic[path] = []
                sanity_pathdic[path]=[]
            else:
                if path.split('-')[0] in track:
                    pathdic[path] = []
                    sanity_pathdic[path]=[]
        
        for each in list(perturbationresultdic.keys()):
            if each != 'backScore':

                for k in list(perturbationresultdic[each].keys()):
                    if k!= 'statistic' and k in pathdic.keys():
                        # pathdic[k].append(perturbationresultdic[each][k][score][0])
                        pathdic[k].append(perturbationresultdic[each][k][score]) #avg score
        for each in list(sanity_perturbationresultdic.keys()):
            if each != 'backScore':
                for k in list(sanity_perturbationresultdic[each].keys()):
                    if k!= 'statistic' and k in sanity_pathdic.keys():
                        # sanity_pathdic[k].append(sanity_perturbationresultdic[each][k][score][0])
                        sanity_pathdic[k].append(sanity_perturbationresultdic[each][k][score])#avg score
    
        return pathdic,sanity_pathdic


    def getTrackPathwayCDF(self,pathway,perturbationresultdic,path,pathdic,sanity_pathdic,gene_in_pathway,scoreindex,flag=0):
        '''
        for this function, only get pathway cdf in a specific path
        
        args:
        pathway: the pathway to get cdf
        path: path to get pathway cdf 
        perturbationresultdic: perturbation statistic results
        pathdic: scores of pathways in every path
        sanity_pathdic: sanity (random) scores of pathways in every path
        scoreindex: name of score to print
        flag:0: pushback, 1: pushforward
        '''
    
        pathdic[path] = np.array(pathdic[path])
        sanity_pathdic[path]= np.array(sanity_pathdic[path])
    
        score = perturbationresultdic[pathway][path][scoreindex]

        
        cdf=norm.cdf(perturbationresultdic[pathway][path][scoreindex],sanity_pathdic[path].mean(),sanity_pathdic[path].std())    
        return score, 1.0000000-cdf
    def getSummarizedResults(self,track_to_analysis,pathwayranking, pathwaydic,track_percentage,gene_in_pathway,scoreindex,direction_dict,all):#(pathwayranking,perturbationresultdic,pathdic,sanity_pathdic,gene_in_pathway,score):
        '''
        args:
        pathwayranking: ordered pathway based on scores (both top and down)
        pathwaydic: pathway dictionary, distance info
        gene_len_dict: number of gene in each pathway
        sanity_directory: directory to sanity results
        gene_in_pathway:  genes in each pathway overlapping with adatagene
        scoreindex: perturbation score
        
        '''

        sanity_pathwaydic = self.load(gene_in_pathway,track_percentage,all=all,sanity=True)
        # np.save('sanity_pathwaydic.npy',sanity_pathwaydic)
        # np.save('pathwaydic.npy',pathwaydic)
        # print('saved')
        # print(gdsg)
        sanity_perturbationresultdic = self.calculateAvg(sanity_pathwaydic,sanity=True,tracklist=track_to_analysis)
        perturbationresultdic = self.calculateAvg(pathwaydic,tracklist=track_to_analysis)
        pathways = list(pathwaydic.keys())
        if all:
            pathdic,sanity_pathdic = self.conver_pathway_track_score_to_track_pathway_score(pathways,perturbationresultdic,sanity_perturbationresultdic,scoreindex,'total')
        else:
            pathdic,sanity_pathdic = self.conver_pathway_track_score_to_track_pathway_score(pathways,perturbationresultdic,sanity_perturbationresultdic,scoreindex,track_to_analysis)
        tendency_dict = self.getTendencyFromIDREM()

        tendency = []
        for each in list(track_percentage.keys()):
            each1 = each.split('-')[-1]
            if len(tendency) == 0:
                tendency = np.array(tendency_dict[each1])*track_percentage[each]
            else:
                tendency = tendency + np.array(tendency_dict[each1])*track_percentage[each]
        infodict = {}
        pathwayranking = dict(pathwayranking)
        for outer, inner in pathwayranking.keys():
            if outer not in infodict.keys():
                infodict[outer] = {}
            infodict[outer][inner] = {}
            infodict[outer][inner]['compound'] = {}
            for idx,each in enumerate(list(pathwayranking[(outer,inner)])):
                infodict[outer][inner]['compound'][str(idx)]=each

        for track in list(infodict.keys()):
            # del infodict[track]['down_compounds']
            for updown in list(infodict[track].keys()):

                infodict[track][updown]['perturbation score'] = {}
                infodict[track][updown]['pval_adjusted'] = {}
                infodict[track][updown]['drug_regulation'] = {}
                infodict[track][updown]['idrem_suggestion'] = {}
                count = 0
                for idx in range(len(infodict[track][updown]['compound'].keys())):
                    eachpathway = infodict[track][updown]['compound'][str(idx)]
                    if eachpathway == 'backScore':
                        continue
                    if str(eachpathway) == 'nan':
                        continue
                    count+=1
                
                for idx in range(len(infodict[track][updown]['compound'].keys())):
                    eachpathway = infodict[track][updown]['compound'][str(idx)]

                    if eachpathway == 'backScore':
                        continue
                    if str(eachpathway) == 'nan':
                        continue
                    flag = 0
                    if updown == 'down_compounds':
                        flag = 1
                
                    score, p=self.getTrackPathwayCDF(eachpathway,perturbationresultdic,track,pathdic, sanity_pathdic, gene_in_pathway, scoreindex,flag=flag)
                    p=p*count/(count-idx)
                    if p >1:
                        p = 1
                    infodict[track][updown]['perturbation score'][str(idx)] =score
                    infodict[track][updown]['pval_adjusted'][str(idx)] = p
                    infodict[track][updown]['drug_regulation'][str(idx)] = gene_in_pathway[eachpathway]
                    written_gene = []
                    for eachgene in gene_in_pathway[eachpathway]:
                        eachgene = eachgene.split(':')[0]
                        temp = direction_dict[eachgene]
                        if temp > 0: #increasing tendency decrease gene expression
                            written_gene.append(eachgene+':-')
                        else: #decreasing tendency increase gene expression
                            written_gene.append(eachgene+':+')
                    infodict[track][updown]['idrem_suggestion'][str(idx)] = written_gene
        # infodict = pd.json_normalize(infodict)
        # infodict.columns = infodict.columns.str.split(".").map(tuple)
        # infodict = infodict.stack([0, 1, 2]).reset_index(0, drop=True)
        # infodict = infodict.transpose()
        # infodict = infodict.reset_index()
        # infodict['index'] = infodict['index'].astype('int32')
        # infodict = infodict.set_index(['index']).sort_index()
        return infodict
    def getTendencyDict(self,track_percentage):
        genenames = self.adata.var.index.tolist()
        tendency_dict = self.getTendencyFromIDREM()
        tendency = None
        for each in list(track_percentage.keys()):
            each1 = each.split('-')[-1]
            if tendency is None:
                tendency = np.array(tendency_dict[each1])*track_percentage[each]
            else:
                tendency = tendency + np.array(tendency_dict[each1])*track_percentage[each]
        output = {}
        for i, each in enumerate(genenames):
            if  each not in output.keys():
                output[each] = tendency[i]
        return output
    def get_online_results(self, score, random_score):
        cdf=norm.cdf(score,np.array(random_score).mean(),np.array(random_score).std())
        return 1.00 - cdf
#load avg needed

    def main_analysis(self,track_to_analysis, all,score=None,items=None,custommode=None):
        '''
        all: whether to put all tracks together
        '''

        tracks = self.get_tracks()
        track_percentage = self.get_track_percentage(tracks)
        if self.mode == 'pathway':
            items = self.adata.uns['data_pathway_overlap_genes']
            #  items=dict(np.load('./iterativeTrainingNOV26/4/data_pathway_overlap_genes.npy',allow_pickle=True).tolist())#together up and down
        else:
            items = self.adata.uns['data_drug_overlap_genes']
            #=dict(np.load('./iterativeTrainingNOV26/4/dec23_drug_target_with_direction.npy',allow_pickle=True).tolist())
        # if items is None:
        #     items=dict(np.load('./iterativeTrainingNOV26/4/dec23_drug_target_with_direction.npy',allow_pickle=True).tolist())#together up and down
        # else:
        #     items=dict(np.load(items,allow_pickle=True).tolist())
        direction_dict = self.getTendencyDict(track_percentage)
        pathwaydic = self.load(items,track_percentage,all=all)
        
        topdownpathways = self.getTopDownPathways(pathwaydic,track_percentage,track_to_analysis,all=all)
        return self.getSummarizedResults(track_to_analysis,topdownpathways,pathwaydic,track_percentage,items,score,direction_dict,all=all)
    def online_analysis(self,deltaD):
        import time

        self.adata.obs['stage'] = self.adata.obs['stage'].astype('string')
        if self.allTracks == True:
            tracks = self.get_tracks()
            
            track_percentage = self.get_track_percentage(tracks)
            
            
            perturbation_scores = self.load_online(deltaD,track_percentage=track_percentage)
            
            random_scores = self.load_online(sanity=True,track_percentage=track_percentage)
            
        elif self.allTracks == False:
            track = deltaD[0]
            perturbation_scores =  self.load_online(deltaD[1]) # if one track mode, delta = [track,[deltaDA,deltaDB]]
            random_scores =self.load_online(sanity=True)
        
       
        perturbation_score = np.array(list(perturbation_scores.values()))#.mean()
        random_score = np.array(list(random_scores.values()))
        
        pval = self.get_online_results(perturbation_score, random_score)
        
        
        print(perturbation_score, pval, self.out_delta)
        
        return perturbation_score, pval, self.out_delta
