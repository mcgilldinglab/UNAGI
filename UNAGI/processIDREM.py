import numpy as np
import gc
import os
import anndata
import pandas as pd
import scanpy as sc
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import KernelDensity
from sklearn import cluster
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, entropy, multivariate_normal, gamma
from scipy import stats 
import torch
from torch.nn import functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.distributions.gamma import Gamma
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.stats import entropy 
from scipy import stats
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import threading
import subprocess
from scipy.stats import multivariate_normal
import gc
def getClusterPaths(edges):
    '''
    obtain the paths of each clusters
    args:
    edeges: contains three lists of edges between control group and IPF 1 stage, IPF 1 stage and IPF 2 stage, IPF 2 stage and IPF 3 stage
    
    return:
    paths: a collection of paths of clusters
    '''
    paths = {}
    C2oneEdge = edges[0]
    one2twoEdge = edges[1]
    two2threeEdge = edges[2]
    for each in C2oneEdge:
        if str(each[0]) not in paths.keys():
            paths[str(each[0])]=[[each[0]],[each[1]]]
        else:
            paths[str(each[0])][1].append(each[1])

    #connect2 = {}
    for each in one2twoEdge:
        for item in paths.keys():
            if each[0] in paths[item][1]:
                if len(paths[item]) == 2:
                    paths[item].append([each[1]])
                else:
                    paths[item][2].append(each[1])
                

    for each in two2threeEdge:
        for item in paths.keys():
            if len(paths[item]) == 2:
                continue
            if each[0] in paths[item][2]:
                if len(paths[item]) == 3:
                    paths[item].append([each[1]])
                else:
                    paths[item][3].append(each[1])  
    return paths

def getClusterIdrem(paths,state):
    '''
    concatenate the average gene expression in a cluster tree. shape: [number of stages, number of gene]
    args: 
    paths: the collection of paths
    state: a list of average gene expression of each state
    
    return: 
    out: a list of gene expression of each cluster tree
    '''
    out = []
    for i,each in enumerate(paths.keys()):
        if len(paths[each]) == 4:
            stages = []
            for j, item in enumerate(paths[each]):
                stages.append(averageNode(item,state[j]))
            joint_matrix = np.concatenate((stages[0].reshape(-1,1), stages[1].reshape(-1,1), stages[2].reshape(-1,1), stages[3].reshape(-1,1)),axis =1)
            out.append(joint_matrix)
    return out
def getIdrem(paths,state):
    '''
    concatenate the average gene expression of clusters in each path. shape: [number of stages, number of gene]
    args: 
    paths: the collection of paths
    state: a list of average gene expression of each state
    
    return: 
    out: a list of gene expression of each path
    '''
    out = []
    for i,each in enumerate(paths):
        joint_matrix = np.concatenate((state[0][each[0]].reshape(-1,1), state[1][each[1]].reshape(-1,1), state[2][each[2]].reshape(-1,1), state[3][each[3]].reshape(-1,1)),axis =1)
        out.append(joint_matrix)
    return out

class IDREMthread(threading.Thread):
    def __init__(self, indir, outdir, each,idrem_dir):
        threading.Thread.__init__(self)
        self.indir = indir
        self.outdir = outdir
        self.each = each
        self.idrem_dir = idrem_dir
    def run(self):
        
        command = 'cd '+ str(self.idrem_dir)+' && java -Xmx8G -jar idrem.jar -b %s %s'%(self.indir, self.outdir)
        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        print(p.stdout.read())

def runIdrem(paths, midpath, idremInput, savedPath,genenames,iteration, idrem_dir, trained=False):
    '''
    train IDREM model and save the results in iterative training with midpath and iteration
    args:
    paths: the path of IPF progression
    idremInput: average gene expression of each path
    trained: if the model is trained, use saved model
    
    
    
    '''
    dir1 = os.path.join(midpath, str(iteration)+'/idremInput')
    dir2 = os.path.join(midpath, str(iteration)+'/idremsetting')
    dir3 = os.path.join(midpath, str(iteration)+'/idremModel')

    initalcommand = 'mkdir '+dir1+' && mkdir '+dir2+' && mkdir '+dir3
    p = subprocess.Popen(initalcommand, stdout=subprocess.PIPE, shell=True)
    print(p.stdout.read()) 
    print(paths)
    for i, each in enumerate(paths):
        each_processed = []
        for e in each:
            e = str(e).strip('[]')
            e = e.replace(', ','n')
            each_processed.append(e)
        # each_processed = [str(e).strip('[]').replace(', ', 'n') for e in each]
        print(each_processed)
        file_name = '-'.join(each_processed)
        file_path = os.path.join(midpath, str(iteration), 'idremInput', f'{file_name}.txt')
        header = ['gene'] + [f'stage{j}' for j in range(len(each))]
        with open(file_path, 'w') as f:
            f.write('\t'.join(header) + '\n')
            for j, row in enumerate(idremInput[i]):
                row_data = '\t'.join(str(r) for r in row)
                f.write("%s\t%s\n" % (genenames[j], row_data))
        examplefile = open(os.path.join(idrem_dir, 'example_settings.txt'), 'r')
        settings_file_path = os.path.join(midpath, str(iteration), 'idremsetting', f'{file_name}.txt')
        with open(settings_file_path, 'w') as f:
            for k, line in enumerate(examplefile.readlines()):

                if trained and k == 4:
                    print(midpath) 
                    f.write('%s\t%s\n' % ('Saved_Model_File', os.path.join(os.path.abspath(os.path.join(midpath, str(iteration), 'idremInput')), f'{file_name}.txt')))
                elif k == 1:
                    f.write('%s\t%s\n' % ('TF-gene_Interaction_Source', 'human_encode.txt.gz'))
                    continue
                elif k == 2:
                    f.write('%s\t%s\n' % ('TF-gene_Interactions_File', 'TFInput/human_encode.txt.gz'))
                    continue
                elif k == 5:
                    f.write('%s\t%s\n' % ('Gene_Annotation_Source', 'Human (EBI)'))
                    continue
                elif k == 6:
                    f.write('%s\t%s\n' % ('Gene_Annotation_File', 'goa_human.gaf.gz'))
                    continue 
                elif k== 17:
                    f.write('%s\n' % ('miRNA-gene_Interaction_Source'))
                    continue
                elif k== 18:
                    f.write('%s\n' % ('miRNA_Expression_Data_File'))
                    continue
                elif k== 26:
                    f.write('%s\n' % ('Proteomics_File'))
                    continue
                elif k == 34:
                    f.write('%s\n' % ('Epigenomic_File'))
                    continue
                elif k == 35:
                    f.write('%s\n' % ('GTF File'))
                    continue
                elif k == 42:
                    f.write('%s\t%s\n' % ('Minimum_Absolute_Log_Ratio_Expression', '0.05'))
                    continue
                elif k ==51 :
                    f.write('%s\t%s\n' % ('Convergence_Likelihood_%', '0.1'))
                    continue
                elif k == 52:
                    f.write('%s\t%s\n' % ('Minimum_Standard_Deviation', '0.01'))
                    continue
                elif k != 3:
                    f.write(line)
                else:
                    f.write('%s\t%s\n' % ('Expression_Data_File', os.path.join(os.path.abspath(os.path.join(midpath, str(iteration), 'idremInput')), f'{file_name}.txt')))

        examplefile.close()
        # each0 = str(each[0]).strip('[]')
        # each1 = str(each[1]).strip('[]')
        # each2 = str(each[2]).strip('[]')
        # each3 = str(each[3]).strip('[]')
        
        # each0 = each0.replace(', ','n')
        # each1 = each1.replace(', ','n')
        # each2 = each2.replace(', ','n')
        # each3 = each3.replace(', ','n')
        
        # with open(os.path.join(midpath,str(iteration)+'/idremInput/%s-%s-%s-%s.txt'%(each0,each1,each2,each3)), 'w') as f:
        #     f.write("%s\t%s\t%s\t%s\t%s\n"%('gene','control','IPF1','IPF2','IPF3'))
        #     for j,row in enumerate(idremInput[i]):
        #         f.write("%s\t%s\t%s\t%s\t%s\n" % (genenames[j],str(row[0]),str(row[1]), str(row[2]),str(row[3])))   
        # examplefile = open('./idrem-master/example_settings.txt','r')
    
        # with open(os.path.join(midpath,str(iteration)+'/idremsetting/%s-%s-%s-%s.txt'%(each0,each1,each2,each3)), 'w') as f:
        #     for k,line in enumerate(examplefile.readlines()):
        #         if trained == True:
        #             if k == 4:
        #                 print(os.path.join(midpath,str(iteration)+'/idremInput/%s-%s-%s-%s.txt'%(each0,each1,each2,each3)))
        #                 f.write('%s\t%s\n'%('Saved_Model_File',os.path.join('../'+midpath,str(iteration)+'/idremInput/%s-%s-%s-%s.txt'%(each0,each1,each2,each3))))
        #         if k!= 3:
        #             f.write(line)
        #         else:
        #             f.write('%s\t%s\n'%('Expression_Data_File', os.path.join('../'+midpath,str(iteration)+'/idremInput/%s-%s-%s-%s.txt'%(each0,each1,each2,each3))))
    settinglist = os.listdir(os.path.join(midpath,str(iteration)+'/idremsetting/'))
    
    print(settinglist)
    threads = []
    for each in settinglist:
        if each[0] != '.':
            
            indir = os.path.abspath(os.path.join(midpath,str(iteration)+'/idremsetting/',each))
            outdir =os.path.join(os.path.abspath(os.path.join(midpath,str(iteration))+'/idremModel/'),each)
            
            threads.append(IDREMthread(indir, outdir, each,idrem_dir))
    count = 0
    while True:
        if count<len(threads) and count +2 > len(threads):
            threads[count].start()
            threads[count].join()
            break
        elif count == len(threads):
            break
        else:
            threads[count].start()
            threads[count+1].start()
            threads[count].join()
            threads[count+1].join()
            count+=2
    if not trained:
        print(os.getcwd())
        dir1 = os.path.join(midpath, str(iteration)+'/idremResults')
        dir2 = os.path.join(midpath, str(iteration)+'/idremInput/*.txt_viz')
        command = [['rm -r '+dir1],[ ' mkdir '+dir1], [' mv '+dir2+ ' '+dir1]]
        for each in command:
            p = subprocess.Popen(each, stdout=subprocess.PIPE, shell=True)
            print(p.stdout.read())
    print('idrem Done')
# def runIdrem(paths, idremInput, savedPath,genenames, trained=False):
#     '''
#     train IDREM model and save the results 
#     args:
#     paths: the path of IPF progression
#     idremInput: average gene expression of each path
#     trained: if the model is trained, use saved model
    
    
    
#     '''
#     initalcommand = 'rm -r ./reresult/idremInput/ && rm -r ./reresult/idremsetting && rm -r ./reresult/idremModel && mkdir ./reresult/idremInput && mkdir ./reresult/idremsetting && mkdir ./reresult/idremModel'
#     p = subprocess.Popen(initalcommand, stdout=subprocess.PIPE, shell=True)
#     print(p.stdout.read()) 
#     for i, each in enumerate(paths):
#         each0 = str(each[0]).strip('[]')
#         each1 = str(each[1]).strip('[]')
#         each2 = str(each[2]).strip('[]')
#         each3 = str(each[3]).strip('[]')
#         each0 = each0.replace(', ','n')
#         each1 = each1.replace(', ','n')
#         each2 = each2.replace(', ','n')
#         each3 = each3.replace(', ','n')
#         with open('./reresult/idremInput/%s-%s-%s-%s.txt'%(each0,each1,each2,each3), 'w') as f:
#             f.write("%s\t%s\t%s\t%s\t%s\n"%('gene','control','IPF1','IPF2','IPF3'))
#             for j,row in enumerate(idremInput[i]):
                
#                 f.write("%s\t%s\t%s\t%s\t%s\n" % (genenames[j],str(row[0]),str(row[1]), str(row[2]),str(row[3])))   
#         examplefile = open('./idrem-master/example_settings.txt','r')
    
#         with open('./reresult/idremsetting/%s-%s-%s-%s.txt'%(each0,each1,each2,each3), 'w') as f:
#             for k,line in enumerate(examplefile.readlines()):
#                 if trained == True:
#                     if k == 4:
#                         f.write('%s\t%s\n'%('Saved_Model_File','../reresult/idremInput/%s-%s-%s-%s.txt'%(each0,each1,each2,each3)))
#                 if k!= 3:
#                     f.write(line)
#                 else:
#                     f.write('%s\t%s\n'%('Expression_Data_File', '../reresult/idremInput/%s-%s-%s-%s.txt'%(each0,each1,each2,each3)))
#     settinglist = os.listdir('./reresult/idremsetting/')
    
#     print(settinglist)
#     threads = []
#     for each in settinglist:
#         if each[0] != '.':
#             indir = os.path.join('../reresult/idremsetting/',each)
#             outdir = os.path.join('../reresult/idremModel/',each)
#             threads.append(IDREMthread(indir, outdir, each))
#     count = 0
#     while True:
#         if count<len(threads) and count +2 > len(threads):
#             threads[count].start()
#             threads[count].join()
#             break
#         elif count == len(threads):
#             break
#         else:
#             threads[count].start()
#             threads[count+1].start()
#             threads[count].join()
#             threads[count+1].join()
#             count+=2
#     if not trained:
#         command = 'rm -r ./reresult/'+ 'idremResults0.1-nov14' + ' && mkdir ./reresult/'+'idremResults0.1-nov14'+' && mv ./reresult/idremInput/*.txt_viz ./reresult/'+'idremResults0.1-nov14'+'/'
#         p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
#         print(p.stdout.read())
#         print('idrem Done')
def averageNode(nodes,state):
    '''
    calculate the average gene expression of sibling nodes
    args: 
    nodes: number of sibling nodes
    state: the gene expression of each cluster in a certain stage
    
    return: 
    out: the average gene expression of sibling nodes
    '''
    out = 0
    for each in nodes:
        out+=state[each]
    return out/len(nodes)
# def runIDREMSettings(midpath,iteration):
#     settinglist = os.listdir('./'+midpath+'/'+str(iteration)+'/idremsetting/')
    
#     print(settinglist)
#     threads = []
#     for each in settinglist:
#         if each[0] != '.':
#             indir = os.path.join('../'+midpath+'/'+str(iteration)+'/idremsetting/',each)
#             outdir = os.path.join('../'+midpath+'/'+str(iteration)+'/idremModel/',each)
#             threads.append(IDREMthread(indir, outdir, each))
#     count = 0
#     while True:
#         if count<len(threads) and count +2 > len(threads):
#             threads[count].start()
#             threads[count].join()
#             break
#         elif count == len(threads):
#             break
#         else:
#             threads[count].start()
#             threads[count+1].start()
#             threads[count].join()
#             threads[count+1].join()
#             count+=2

#     command = [['rm -r ./'+midpath+'/'+str(iteration)+'/idremResults'],[ 'mkdir ./'+midpath+'/'+str(iteration)+'/idremResults/'], ['mv ./'+midpath+'/'+str(iteration)+'/idremInput/*.txt_viz ./'+midpath+'/'+str(iteration)+'/idremResults/']]
#     for each in command:
#         p = subprocess.Popen(each, stdout=subprocess.PIPE, shell=True)
#         print(p.stdout.read())
#     print('idrem Done')

