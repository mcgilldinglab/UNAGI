import numpy as np
import os
import threading
import subprocess
import gc
def getClusterPaths(edges, total_stages):
    '''
    Obtain the paths of each cluster for multiple stages.
    
    parameters
    -----------
    edges: list
        A list of lists, where each sublist contains edges between consecutive stages.
    total_stages: int
        Total number of stages.

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
            paths[str(each[0])] = [[each[0]], [each[1]]]
        else:
            paths[str(each[0])][1].append(each[1])

    # Iterate through remaining stages
    for stage in range(1, total_stages - 1):
        for each in edges[stage]:
            for item in paths.keys():
                if len(paths[item]) == stage:
                    continue
                if each[0] in paths[item][stage]:
                    if len(paths[item]) == stage + 1:
                        paths[item].append([each[1]])
                    else:
                        paths[item][stage + 1].append(each[1])

    return paths

def getClusterIdrem(paths, state, total_stages):
    '''
    Concatenate the average gene expression in a cluster tree. Shape: [number of stages, number of genes]
    
    parameters
    -----------
    paths: The collection of paths.
    state: A list of average gene expression of each state.
    total_stages: Total number of stages.
    
    return
    -----------
    out: A list of gene expression of each cluster tree.
    '''
    out = []

    for path_key in paths.keys():
        path = paths[path_key]

        # Ensure the path contains the expected number of stages
        if len(path) == total_stages:
            stages = [averageNode(node, state[i]) for i, node in enumerate(path)]

            # Reshape each stage and concatenate
            reshaped_stages = [stage.reshape(-1, 1) for stage in stages]
            joint_matrix = np.concatenate(reshaped_stages, axis=1)

            out.append(joint_matrix)

    return out

def getIdrem(paths,state):
    '''
    concatenate the average gene expression of clusters in each path. shape: [number of stages, number of gene]
    parameters
    ----------------------
    paths: list
        the list of paths
    state: list
        a list of average gene expression of each state
    
    return
    ---------------------- 
    out: list
        a list of gene expression of each path
    '''
    out = []
    for i,each in enumerate(paths):
        joint_matrix = np.concatenate((state[0][each[0]].reshape(-1,1), state[1][each[1]].reshape(-1,1), state[2][each[2]].reshape(-1,1), state[3][each[3]].reshape(-1,1)),axis =1)
        out.append(joint_matrix)
    return out

class IDREMthread(threading.Thread):
    '''
    the thread for running IDREM. Support multiple threads.
    '''
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

def runIdrem(paths, midpath, idremInput,genenames,iteration, idrem_dir, species='Human', Minimum_Standard_Deviation = 0.01,Convergence_Likelihood=0.1,Minimum_Absolute_Log_Ratio_Expression=0.05, trained=False):
    '''
    train IDREM model and save the results in iterative training with midpath and iteration
    
    parameters
    ----------------------
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
        examplefile_path = os.path.join(idrem_dir, 'example_settings.txt')#open(os.path.join(idrem_dir, 'example_settings.txt'), 'r')
        settings_file_path = os.path.join(midpath, str(iteration), 'idremsetting', f'{file_name}.txt')
        with open(examplefile_path, 'r') as examplefile:
        # Open settings_file_path for writing
            with open(settings_file_path, 'w') as f:
                for k, line in enumerate(examplefile.readlines()):

                    if trained and k == 4:
                        print(midpath) 
                        f.write('%s\t%s\n' % ('Saved_Model_File', os.path.join(os.path.abspath(os.path.join(midpath, str(iteration), 'idremInput')), f'{file_name}.txt')))
                    elif k == 1:
                        if species == 'Human':
                            f.write('%s\t%s\n' % ('TF-gene_Interaction_Source', 'human_encode.txt.gz'))
                            
                            continue
                    elif k == 2:
                        if species == 'Human':
                            f.write('%s\t%s\n' % ('TF-gene_Interactions_File', 'TFInput/human_encode.txt.gz'))
                            continue
                    elif k == 5:
                        if species == 'Human':
                            f.write('%s\t%s\n' % ('Gene_Annotation_Source', 'Human (EBI)'))
                            continue
                    elif k == 6:
                        if species == 'Human':
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
                        f.write('%s\t%s\n' % ('Minimum_Absolute_Log_Ratio_Expression', str(Minimum_Absolute_Log_Ratio_Expression)))
                        continue
                    elif k ==51 :
                        f.write('%s\t%s\n' % ('Convergence_Likelihood_%', str(Convergence_Likelihood)))
                        continue
                    elif k == 52:
                        f.write('%s\t%s\n' % ('Minimum_Standard_Deviation', str(Minimum_Standard_Deviation)))
                        continue
                    elif k != 3:
                        f.write(line)
                    else:
                        f.write('%s\t%s\n' % ('Expression_Data_File', os.path.join(os.path.abspath(os.path.join(midpath, str(iteration), 'idremInput')), f'{file_name}.txt')))

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

def averageNode(nodes,state):
    '''
    calculate the average gene expression of sibling nodes
    
    parameters
    ----------------------
    nodes: int
        number of sibling nodes
    state: list
        the gene expression of each cluster in a certain stage
    
    return
    -----------
    out: the average gene expression of sibling nodes
    '''
    out = 0
    for each in nodes:
        out+=state[each]
    return out/len(nodes)

if __name__ == '__main__':
    import numpy as np
    edges = [[[2, 0], [0, 1], [8, 3], [3, 4], [3, 5], [4, 6], [5, 7], [6, 8], [1, 9]],  [[1, 0], [4, 1], [3, 2], [0, 3], [7, 5], [0, 6], [8, 7], [10, 10], [3, 11], [8, 12]],[[1, 0], [0, 1], [1, 2], [3, 3], [5, 4], [0, 5], [2, 6], [7, 8], [5, 9], [12, 10], [2, 12]]]
    paths = getClusterPaths(edges,4)
    averageValues = np.load('/mnt/md0/yumin/to_upload/UNAGI/UNAGI/data/example/2/averageValues.npy',allow_pickle=True)
    # print(averageValues.shape)
    idrem= getClusterIdrem(paths,averageValues,4)
    print(idrem)