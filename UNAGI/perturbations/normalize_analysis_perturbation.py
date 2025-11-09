'''
This module analyses the perturbation results. It contains the main function to calculate the perturbation score and calculate the p-values.
'''
import csv
import json
import os
from turtle import distance
import scanpy as sc
import numpy as np
import pandas as pd
from scipy.stats import norm

def get_all_distance_changes(distance_change_dict1,distance_change_dict2):
    total_distance_changes = []
    for item in distance_change_dict1.keys():
        for each_track in distance_change_dict1[item].keys():
            for i in range(len(np.array(list(distance_change_dict1[item][each_track])))):
                temp = np.array(list(distance_change_dict1[item][each_track]))[i]
                temp2 = np.array(list(distance_change_dict2[item][each_track]))[i]
                for each_stage in range(len(temp)):
                    if each_stage != i:
                        total_distance_changes.append(temp[each_stage])
                        total_distance_changes.append(temp2[each_stage])
    return total_distance_changes, np.array(total_distance_changes).mean(), np.array(total_distance_changes).std()

class perturbationAnalysis:
    '''
    The perturbationAnalysis class takes the adata object and the directory of the task as the input. 

    parameters
    -----------
    adata: AnnData object
        The adata object contains the single-cell data.
    target_directory: str
        The directory of the task.
    log2fc: float
        The log2 fold change of the perturbation. 
    stage: int
        The stage of the time-series single-cell data to analyze the perturbation results. If the stage is None, the perturbation analysis will be performed on all stages.
    mode: str
        The mode of the perturbation. The mode can be either 'pathway', 'online', or 'compound'. 
    allTracks: bool
        If allTracks is True, the perturbation analysis will be performed on all tracks. Otherwise, the perturbation analysis will be performed on a single track.

    '''
    def __init__(self,adata,target_directory,log2fc,stage=None,mode=None,allTracks=None):
        self.adata = adata
        self.log2fc = log2fc
        self.adata.var.index = self.adata.var.index.str.upper()
        self.mode = mode
        self.target_directory = target_directory
        self.stage = stage
        self.allTracks = allTracks
        self.idrem_path = self.target_directory#os.path.join(self.target_directory,'idremVizCluster/')#'./'+ self.target_directory +'/idremVizCluster/'
    def readIdremJson(self, filename):
        '''
        Parse the IDREM json file.

        parameters
        -----------
        filename: str
            The name of the IDREM json file.

        return
        --------
        tt: list
            The parsed the IDREM results.
        '''

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
        
        parameters
        -----------
        filename: str
            the file path of IDREM results
        
        return
        --------
        out: list
            A list of tendency of each path
        '''
        out = []
        tt = self.readIdremJson(filename)
        total_stages = len(filename.split('.')[0].split('-'))
        temp = np.array(tt[8])
        idrem_genes = temp[1:,0].tolist()
        tendency = (temp[1:,-1].astype(float)- temp[1:,1].astype(float))/(total_stages-1)
        # stage_expression = temp[1:,1:5].astype(float)
        return tendency

    def getTendencyFromIDREM(self):
        '''
        get the tendency of each path from iDREM results
        
        parameters
        ------------
        None

        return
        ---------
        out: list
            a list of tendency of each path
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
        '''
        Get all the tracks from the dataset

        parameters
        -----------
        None

        return
        -----------
        tracks: list
            A list of tracks.
        '''
        if 'online_random_background_perturbation_deltaD' in list(self.adata.uns.keys()):
            tracks = self.adata.uns['online_random_background_perturbation_deltaD']['A'].keys()
        else:
            tracks = self.adata.uns['%s_perturbation_deltaD'%self.mode][str(self.log2fc)].keys()

        return list(set(tracks))
    
    
    def load(self,data_pathway_overlap_genes,track_percentage,score_weight,overall_perturbation_analysis=True,sanity=False,mean_distance_change=None, std_distance_change=None):
        '''
        Load the perturbation results and calculate the perturbation scores for each track or for the whole dataset. 

        parameters
        -----------
        data_pathway_overlap_genes: dict
            The pathway overlap genes.
        track_percentage: dict
            The percentage of cells in each track. If track_percentage is None, the perturbation scores will be calculated for each track. Otherwise, the perturbation scores will be calculated for the whole dataset.
        overall_perturbation_analysis: bool
            If overall_perturbation_analysis is True, the perturbation scores will be calculated for the whole dataset. Otherwise, the perturbation scores will be calculated for each track.
        sanity: bool
            If sanity is True, the perturbation results are from the random background perturbation. Otherwise, the perturbation results are from the in-silico perturbation.

        return
        ----------
        out: dict
            The perturbation scores.

        '''
        if sanity == True:
            k1 = self.adata.uns['random_background_perturbation_deltaD'][str(self.log2fc)]
            k2 = self.adata.uns['random_background_perturbation_deltaD'][str(1/self.log2fc)]
            records_to_adjust_weight = []
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
                    
                    if overall_perturbation_analysis == True:
                        if each not in track_percentage.keys():
                            continue
                        pathwaydic1[each1][each] = [np.array(k1[each][each1][i]) for i in range(total_len)]
                    else:
                        pathwaydic1[each1][each] = [np.array(k1[each][each1][i]) for i in range(total_len)]
  
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
                    if overall_perturbation_analysis == True:
                        if each not in track_percentage.keys():
                            continue
                        pathwaydic2[each1][each]= [np.array(k2[each][each1][i]) for i in range(total_len)]
                    else:
                        pathwaydic2[each1][each]= [np.array(k2[each][each1][i]) for i in range(total_len)]
        if sanity == True:
            _, mean_distance_change, std_distance_change = get_all_distance_changes(pathwaydic1,pathwaydic2)
        if overall_perturbation_analysis:
            out = {}
            for each in list(pathwaydic1.keys()):
                out[each] = {}
                temp1 = []
                temp2 = []
                temp = []
                unit_score = []
                for each_track in list(pathwaydic1[each].keys()):
                    if len(list(pathwaydic1[each][each_track])) == 0:
                        del pathwaydic1[each][each_track]
                        del pathwaydic2[each][each_track]

                    # temp_track_score.append(0)
                    # temp_track_score_2.append(0)
                    temp_track_score = []
                    for i in range(len(np.array(list(pathwaydic1[each][each_track])))):
                        if self.stage is None:
                            temp_track_score.append(np.array(track_percentage[each_track])*np.abs(self.calculateScore(np.array(list(pathwaydic1[each][each_track]))[i],i,mean_distance_change=mean_distance_change,std_distance_change=std_distance_change,weight=score_weight)[0] - self.calculateScore(np.array(list(pathwaydic2[each][each_track]))[i],i,mean_distance_change=mean_distance_change,std_distance_change=std_distance_change,weight=score_weight)[0])/2)
                        else:
                            if i != self.stage:
                                continue
                            elif i == self.stage:
                                temp_track_score.append(np.array(track_percentage[each_track])*np.abs(self.calculateScore(np.array(list(pathwaydic1[each][each_track]))[i],i,mean_distance_change=mean_distance_change,std_distance_change=std_distance_change,weight=score_weight)[0] - self.calculateScore(np.array(list(pathwaydic2[each][each_track]))[i],i,mean_distance_change=mean_distance_change,std_distance_change=std_distance_change,weight=score_weight)[0])/2)
                    unit_score.append(np.mean(temp_track_score))
                # for i in range(len(np.sum(np.array(list(pathwaydic1[each].values())),axis=0))):
                #     if self.stage is None:
                #         temp1.append(self.calculateScore(np.sum(np.array(list(pathwaydic1[each].values())),axis=0)[i],i))
                #         temp2.append(self.calculateScore(np.sum(np.array(list(pathwaydic2[each].values())),axis=0)[i],i))
                #     else:
                #         if i != self.stage:
                #             continue
                #         elif i == self.stage:
                #             temp1.append(self.calculateScore(np.sum(np.array(list(pathwaydic1[each].values())),axis=0)[i],i))
                #             temp2.append(self.calculateScore(np.sum(np.array(list(pathwaydic2[each].values())),axis=0)[i],i))
                # temp1 = np.array(temp1)
                # temp2 = np.array(temp2)

                # temp.append(np.sqrt(((np.abs(np.mean(temp1[:,0]))+np.abs(np.mean(temp2[:,0])))/2) * ((np.abs(np.mean(temp1[:,1]))+np.abs(np.mean(temp2[:,1])))/2)))
                    
                out[each]['overall'] = np.sum(np.array(unit_score))
                if sanity:
         
                    records_to_adjust_weight.append(np.sum(np.array(unit_score)))
        else:
            out = {}
            for each in list(pathwaydic1.keys()): #each is the pathway name
                out[each] = {}
                
                for item in list(pathwaydic1[each].keys()):#item is the track name
                    temp1 = []
                    temp2 = []
                    temp = []

                    for i in range(len(np.array(list(pathwaydic1[each][item])))):
                        if self.stage is None:
                            temp.append(np.abs(self.calculateScore(np.array(list(pathwaydic1[each][item]))[i],i,mean_distance_change=mean_distance_change,std_distance_change=std_distance_change,weight=score_weight)[0] - self.calculateScore(np.array(list(pathwaydic2[each][item]))[i],i,mean_distance_change=mean_distance_change,std_distance_change=std_distance_change,weight=score_weight)[0])/2)
                        else:
                            if i != self.stage:
                                continue
                            else:
                                temp.append(np.abs(self.calculateScore(np.array(list(pathwaydic1[each][item]))[i],i,mean_distance_change=mean_distance_change,std_distance_change=std_distance_change,weight=score_weight)[0] - self.calculateScore(np.array(list(pathwaydic2[each][item]))[i],i,mean_distance_change=mean_distance_change,std_distance_change=std_distance_change,weight=score_weight)[0])/2)
                    temp = np.sum(temp)
                    
                    # temp.append(np.sqrt(((np.abs(np.mean(temp1_copy[:,0]))+np.abs(np.mean(temp2_copy[:,0])))/2) * ((np.abs(np.mean(temp1_copy[:,1]))+np.abs(np.mean(temp2_copy[:,1])))/2)))
                    
                    out[each][item] = np.array(temp)
        if sanity == False:
    
            for each in list(out.keys()):
                if each not in data_pathway_overlap_genes:
                    del out[each]
            return out
        else:
            
            return out, records_to_adjust_weight, mean_distance_change, std_distance_change
    
    #convert distance to scores and some statistics

    def get_cluster_data_size(self,stage,cluster):
        '''
        Get the number of cells in a cluster.

        parameters
        -----------
        stage: int
            The stage of the time-series single-cell data.
        cluster: str
            The cluster id of the selected cluster.

        return
        --------
        cells: int
            The number of cells in the selected cluster.
        '''
        stagedata = self.adata[self.adata.obs['stage'] == str(stage)]#.index.tolist()
        # stagedata = self.adata[stageids]
       
        clusterids = stagedata.obs[stagedata.obs['leiden'] == str(cluster)].index.tolist()
        clusterdata = stagedata[clusterids]
        cells = len(clusterdata)
        return cells
    def get_track_percentage(self, tracks, perturbed_tracks):
        '''
        Get the percentage of the number of cells for each track in the whole dataset.

        parameters
        -----------
        tracks: list
            A list of tracks.
        perturbed_tracks: list
            A list of tracks to analyze. If perturbed_tracks is 'all', then all tracks will be analyzed. Otherwise, only the tracks in perturbed_tracks will be analyzed. e.g. ['0','3','4']

        return
        ----------
        percentage: dict
            The percentage of the number of cells for each track in the whole dataset.
        '''
        percentage = {}
        total_cells = 0
        memory = {}

        stageadatas = [self.adata[self.adata.obs[self.adata.obs['stage'] == stage].index.tolist()] for stage in self.adata.obs['stage'].unique()]
        for each_track in tracks:
           
            if '-' not in each_track:
                continue
            if perturbed_tracks != 'all':
                if each_track.split('-')[0] not in perturbed_tracks:
                    continue
            percentage[each_track] = 0
            clusters = each_track.split('-')
        
            for stage, each_cluster in enumerate(clusters):
                
                cluster_size = stageadatas[stage].obs['leiden'].value_counts().to_dict()[each_cluster]#self.get_cluster_data_size(stageadatas[stage],each_cluster)
                percentage[each_track] += cluster_size
                if stage not in memory.keys():
                    memory[stage] = []
                if each_cluster not in memory[stage]:
                    total_cells+=cluster_size
        for each_track in list(percentage.keys()):
            percentage[each_track] /= total_cells


        return percentage
    def calculateScore(self,delta,flag,mean_distance_change,std_distance_change,weight=1):
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
                each = (each-mean_distance_change)/std_distance_change
           
                out+=each/(2+np.abs(each))
                out1+=np.abs(each/(2+np.abs(each)))
                
   
        return out/(len(delta)-1), out1/(len(delta)-1)

    def calculateAvg(self,pathwaydic,tracklist='all',sanity=False):
        '''
        Calculate the average perturbation score for each track or for the whole dataset.

        parameters
        -----------
        pathwaydic: dict
            The perturbation results.
        tracklist: list
            A list of tracks. If tracklist is 'all', the perturbation scores will be calculated for the whole dataset. Otherwise, the perturbation scores will be calculated for each track.
        sanity: bool
            If sanity is True, the perturbation results are from the random background perturbation. Otherwise, the perturbation results are from the in-silico perturbation.

        return
        --------
        perturbationresultdic: dict
            The perturbation scores.
        '''
        
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
                    if clusterType[0] not in target:
                        if track != 'overall':
                            continue
                perturbationresultdic[each][track] = {}
                perturbationresultdic[each][track]['backScore'] = []
                # for flag, each_perturbation in enumerate(pathwaydic[each][track]):
                #     out = each_perturbation
                #     perturbationresultdic[each][track]['backScore'] = out
                    
                
                perturbationresultdic[each][track]['avg_backScore'] = pathwaydic[each][track]
        return perturbationresultdic



    def getStatistics(self,perturbationresultdic):
        '''
        Get the statistics of the perturbation scores.

        parameters
        -----------
        perturbationresultdic: dict
            The perturbation scores.

        return
        --------
        avg_backScore: list
            The average perturbation scores.
        backScore: list
            The perturbation scores.
        track_name: list
            A list of tracks.
        name_order: list
            The list of objects (compounds or pathways).
        '''
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
        name_order = []
        key.remove('backScore')
        name_order = []
        for i in range(len(backScore)):
            for each in key:
                name_order.append(each)
                backScore[i].append(perturbationresultdic[each][track_name[0]]['backScore'])
                avg_backScore[i].append(perturbationresultdic[each][track_name[0]]['avg_backScore'])
                

        return  avg_backScore,  backScore,track_name,name_order

    def fitlerOutNarrowPathway(self, scores, sanity_scores, names, name_order):
        '''
        Calculate the p-values of the perturbation scores and filter out the ineffective perturbations..
        
        parameters
        ------------
        scores: list
            the score of all perturbations in a track
        sanity_scores: list
            the sanity score of all perturbations in a track


        names: list
            name of perturbation objects (pathways or compounds)
        name_order: list
            the name order of perturbation objects (pathways or compounds)

        return  
        ------------
        top_compounds: list
            the names of top compounds
        down_compounds: list
            the names of down compounds
        '''
   
        
        sanity_scores = np.array(sanity_scores)
        top_compounds = []
        # down_compounds = []
        record_top = []
        # record_down = []

    
        for i,each in enumerate(scores):

            if float(each) >= 0:#sanity_scores.mean()+sanity_scores.std():
                top_compounds.append(names[i])
                record_top.append(i)
            # elif  float(each) <= sanity_scores.mean()-sanity_scores.std():
                
            #     down_compounds.append(names[i])
            #     record_down.append(i)
        scores = np.array(scores)
        # print(down_compounds)
        filtered_top_score = scores[record_top]
        # filtered_down_score = scores[record_down]
        filtered_top_index = np.argsort(filtered_top_score).tolist()
        filtered_top_index.reverse()
        # filtered_down_index = np.argsort(filtered_down_score).tolist()

        filtered_top_score = sorted(filtered_top_score,reverse=True)
        # filtered_down_score = sorted(filtered_down_score)
        
        top_compounds = np.array(top_compounds)
        # down_compounds = np.array(down_compounds)
        top_compounds = top_compounds[filtered_top_index]
        # down_compounds = down_compounds[filtered_down_index]
        
        # final_down_compounds = []
        # final_top_compounds = []
        # for i, each in enumerate(filtered_top_score):
        #     cdf = norm.cdf(each, sanity_scores.mean(),sanity_scores.std())
        #     # if (1.000-cdf) * len(filtered_top_score) / (i + 1)  < 0.05:
        #     if (1.000-cdf) <0.05:
        #         final_top_compounds.append(top_compounds[i])
        # for i, each in enumerate(filtered_down_score):
        #     cdf = norm.cdf(each, sanity_scores.mean(),sanity_scores.std())
        #     if cdf * len(filtered_down_score) / (i + 1) < 0.05:
        #         # print('down compound:',down_compounds[i])
        #         final_down_compounds.append(down_compounds[i])
        return top_compounds#,down_compounds
    def getTopDownObjects(self,pathwaydic,sanity_pathwaydic,track_percentage,track,overall_perturbation_analysis=True, flag=0):
        '''
        get top and down objects in a track

        parameters
        ------------
        pathwaydic: list 
            perturbation statistic results
        sanity_pathwaydic: list
            sanity perturbation statistic results
        track_percentage: list
            percentage of cells in each track
        track: list
            track to analyze
        overall_perturbation_analysis: bool
            if overall_perturbation_analysis is True, analyze all tracks.
        flag: int
            0:both pushback score and pushforward score; 1 only pushback score, -1 only pushforward score

        return
        ------------
        results: dict
            top and down objects in a track
        
        '''

        perturbationresultdic = self.calculateAvg(pathwaydic,tracklist=track)
        sanity_perturbationresultdic = self.calculateAvg(sanity_pathwaydic,sanity=True,tracklist=track)
        

        pathways = list(perturbationresultdic.keys())
        pathways.remove('backScore')

        # pathdic,sanity_pathdic = conver_object_track_score_to_track_object_score(pathways,perturbationresultdic,sanity_perturbationresultdic,scoreindex)
        sanity_avg_backScore, sanity_backScore,sanity_track_name,_ = self.getStatistics(sanity_perturbationresultdic)

        avg_backScore, backScore,track_name,pathway_name_order = self.getStatistics(perturbationresultdic)

        outtext=[]
        results= {}
        for i, each in enumerate(avg_backScore):
            
            results[track_name[i]] = {}
            
            top_compounds=self.fitlerOutNarrowPathway(each,sanity_avg_backScore[i],pathways,pathway_name_order)
            # print(down_compounds)
            results[track_name[i]]['top_compounds']={}
            # results[track_name[i]]['down_compounds']= {}
            for j,each in enumerate(top_compounds):
                results[track_name[i]]['top_compounds'][str(j)] = top_compounds[j]
            # for j,each in enumerate(down_compounds):  
            #     results[track_name[i]]['down_compounds'][str(j)] = down_compounds[j]
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

    def conver_object_track_score_to_track_object_score(self,objects,perturbationresultdic,sanity_perturbationresultdic,score,track):
        '''
        Reorder the dictionary structure of the perturbation scores. The original dictionary structure is object-track-scores. The new dictionary structure is track-object-scores.
       
        
        parameters
        ------------

        objects: list
            A list of names of objects
        perturbationresultdic: list
            A list of perturbation statistic results
        sanity_perturbationresultdic: list
            A list of sanity perturbation statistic results
        score: str
            The type of the perturbation score.
        track: list
            A list of tracks.

        return
        ------------
        pathdic: dict
            The perturbation scores of each track for each object.
        sanity_pathdic: dict
            The sanity perturbation scores of each track for each object.

        '''
        
        pathdic = {}
        sanity_pathdic = {}
        for path in list(perturbationresultdic[objects[0]].keys()):

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


    def getTrackObjectCDF(self,object,perturbationresultdic,path,pathdic,sanity_pathdic,gene_in_object,scoreindex):
        '''
        Calculate the p-values of the perturbation scores of a track for a perturbation object (pathway or compound).

        parameters
        ------------
        object: str
            The name of the perturbation object (pathway or compound).
        perturbationresultdic: dict
            The perturbation scores.
        path: str
            The name of the track.
        pathdic: dict
            The perturbation scores of each track for each object.
        sanity_pathdic: dict
            The sanity perturbation scores of each track for each object.
        gene_in_object: list
            The genes in the perturbation object (pathway or compound).
        scoreindex: str
            The type of the perturbation score.

        return
        ------------
        score: float
            The perturbation score.
        pval: float
            The p-value of the perturbation score.
        '''
    
        pathdic[path] = np.array(pathdic[path])
        sanity_pathdic[path]= np.array(sanity_pathdic[path])
    
        score = perturbationresultdic[object][path][scoreindex]
        # np.save('sanity_pathdic.npy',sanity_pathdic)
        
        cdf=norm.cdf(perturbationresultdic[object][path][scoreindex],sanity_pathdic[path].mean(),sanity_pathdic[path].std())    
        pval = 1.0000000-cdf
        return score, pval
    def getSummarizedResults(self,perturbed_tracks,objectranking, objectdic,sanity_objectdic,track_percentage,gene_in_object,scoreindex,direction_dict,overall_perturbation_analysis):#(pathwayranking,perturbationresultdic,pathdic,sanity_pathdic,gene_in_object,score):
        '''
        Get the perturbation score.

        parameters
        -----------
        perturbed_tracks: list
            A list of tracks to calculate the perturbation scores.
        pathwayranking: list
            The ranked perturbation objects (pathways or compounds).
        objectdic: dict
            The dictionary of the perturbed distance
        track_percentage: dict
            The percentage of number cells of tracks in the whole dataset.
        gene_in_object: dict
            The regulated genes in the perturbation object (pathway or compound).
        scoreindex: str
            The type of the perturbation score.
        direction_dict: dict
            The direction of the gene expression change.
        overall_perturbation_analysis: bool
            If overall_perturbation_analysis is True, the perturbation scores will be calculated for the whole dataset. Otherwise, the perturbation scores will be calculated for each track.

        return
        -----------
        infodict: dict
            The perturbation scores.
        '''

        sanity_perturbationresultdic = self.calculateAvg(sanity_objectdic,sanity=True,tracklist=perturbed_tracks)
        perturbationresultdic = self.calculateAvg(objectdic,tracklist=perturbed_tracks)
        pathways = list(objectdic.keys())
        if overall_perturbation_analysis:
            pathdic,sanity_pathdic = self.conver_object_track_score_to_track_object_score(pathways,perturbationresultdic,sanity_perturbationresultdic,scoreindex,'overall')
        else:
            pathdic,sanity_pathdic = self.conver_object_track_score_to_track_object_score(pathways,perturbationresultdic,sanity_perturbationresultdic,scoreindex,perturbed_tracks)
        tendency_dict = self.getTendencyFromIDREM()

        tendency = []
        for each in list(track_percentage.keys()):
            each1 = each.split('-')[-1]
            if len(tendency) == 0:
                tendency = np.array(tendency_dict[each1])*track_percentage[each]
            else:
                tendency = tendency + np.array(tendency_dict[each1])*track_percentage[each]
        infodict = {}
        objectranking = dict(objectranking)
        for outer, inner in objectranking.keys():
            if outer not in infodict.keys():
                infodict[outer] = {}
            infodict[outer][inner] = {}
            infodict[outer][inner]['compound'] = {}
            for idx,each in enumerate(list(objectranking[(outer,inner)])):
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
                
                    score, p=self.getTrackObjectCDF(eachpathway,perturbationresultdic,track,pathdic, sanity_pathdic, gene_in_object, scoreindex)
                    # p=p*count/(count-idx)
                    # if p >1:
                    #     p = 1
                    infodict[track][updown]['perturbation score'][str(idx)] =score
                    infodict[track][updown]['pval_adjusted'][str(idx)] = p
                    infodict[track][updown]['drug_regulation'][str(idx)] = gene_in_object[eachpathway]
                    written_gene = []
                    for eachgene in gene_in_object[eachpathway]:
                        eachgene = eachgene.split(':')[0]
                        eachgene = eachgene.upper()
                        temp = direction_dict[eachgene]
                        if temp > 0: #increasing tendency decrease gene expression
                            written_gene.append(eachgene+':-')
                        else: #decreasing tendency increase gene expression
                            written_gene.append(eachgene+':+')
                    infodict[track][updown]['idrem_suggestion'][str(idx)] = written_gene
                # adjust p values
                # from statsmodels.stats.multitest import multipletests
                # pvals = list(infodict[track][updown]['pval_adjusted'].values())
                # if len(pvals) > 0:
                #     adjusted_pvals = multipletests(pvals, method='fdr_bh')[1]
                #     for i, each in enumerate(adjusted_pvals):
                #         infodict[track][updown]['pval_adjusted'][str(i)] = each
        return infodict
    def getTendencyDict(self,track_percentage):
        '''
        Get the tendency of the gene expression change.

        parameters
        -----------
        track_percentage: dict
            The percentage of the number of cells of each track in the whole dataset.

        return
        -----------
        output: dict
            The tendency of the gene expression change.
        '''
        genenames = self.adata.var.index.tolist()
        genenames = [each.upper() for each in genenames]
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


    def main_analysis(self,perturbed_tracks, overall_perturbation_analysis,score=None,items=None):

        '''
        The main function to analyse the perturbation results.

        parameters
        -----------
        perturbed_tracks: list
            A list of tracks to calculate the perturbation scores.
        overall_perturbation_analysis: bool
            If overall_perturbation_analysis is True, the perturbation scores will be calculated for the whole dataset. Otherwise, the perturbation scores will be calculated for each track.
        score: str
            The type of the perturbation score.
        items: list
            A list of perturbation objects (pathways or compounds).

        return
        -----------
        results: dict
            The perturbation scores.
        '''

        tracks = self.get_tracks()
        track_percentage = self.get_track_percentage(tracks,perturbed_tracks)
        if self.mode == 'pathway':
            items = self.adata.uns['data_pathway_overlap_genes']
            
        else:
            items = self.adata.uns['data_drug_overlap_genes']

        direction_dict = self.getTendencyDict(track_percentage)
        candidates_wegiht = [0.001,0.01,0.1,0.5,1,5, 10,50,100,500,1000]
        #find a better range of candidates_wegiht
        
        sanity_scores = []
        sanity_pathwaydics = {}
        for weight in candidates_wegiht:
            sanity_pathwaydic,temp_sanity_score = self.load( None ,track_percentage,weight,overall_perturbation_analysis=overall_perturbation_analysis,sanity=True)
            sanity_scores.append(temp_sanity_score)
            sanity_pathwaydics[weight] = sanity_pathwaydic
        sanity_scores = np.array(sanity_scores)
        # differences = np.abs(sanity_scores-0.05) + np.abs(sanity_scores-0.1)
        # differences = np.mean(differences,axis=1)
        # # print('differences:',differences)
        # for i, each in enumerate(np.mean(sanity_scores,axis=1)):
        #     if each > 0.3:
        #         differences[i] = 1000

        # if np.all(differences == 1000):
        #     differences = np.mean(sanity_scores,axis=1)
        min_index = np.argmin(np.abs(np.percentile(sanity_scores,95,axis=1)-0.2))
        weight = candidates_wegiht[min_index]
        print('score weight:',weight)
        sanity_pathwaydic = sanity_pathwaydics[weight]
        objectdic = self.load(items,track_percentage,weight,overall_perturbation_analysis=overall_perturbation_analysis)
        # print('loaded objectdic')
        topdownpathways = self.getTopDownObjects(objectdic,sanity_pathwaydic,track_percentage,perturbed_tracks,overall_perturbation_analysis=overall_perturbation_analysis)
        
        # print('got topdownpathways')
        results = self.getSummarizedResults(perturbed_tracks,topdownpathways,objectdic,sanity_pathwaydic,track_percentage,items,score,direction_dict,overall_perturbation_analysis=overall_perturbation_analysis)
        return  results
   