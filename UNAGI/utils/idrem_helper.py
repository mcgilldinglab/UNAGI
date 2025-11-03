import json
import os
import subprocess
import shutil
import threading
from pathlib import Path
def readIdremJson(path, filename):
    '''
    Parse the IDREM json file
    parameters
    -----------
    path: the file path of IDREM results
    filename: the file name of IDREM results

    return
    -----------
    tt: the parsed IDREM json file
    '''
    print('getting Target genes from ', filename)
    path = os.path.join(path,filename,'DREM.json')
    f=open(path,"r")
    lf=f.readlines()
    f.close()
    lf="".join(lf)
    lf=lf[5:-2]+']'
    tt=json.loads(lf,strict=False)
    return tt

def test_idrem_results(path, filename):
    '''
    Test the branches in the IDREM json file
    
    parameters
    -----------
    path: the file path of IDREM results
    filename: the file name of IDREM results

    return
    -----------
    tt: the parsed IDREM json file
    '''
    if not os.path.exists(os.path.join(path, filename)):
        return False
    json_files = readIdremJson(path, filename)

    total_stages = len(filename.split('-'))
    if len(json_files[0]) > total_stages:
        return True
    else:
        return False

class IDREMthread(threading.Thread):
    '''
    the thread for running IDREM. Support multiple threads.
    '''
    def __init__(self, indir, outdir,idrem_dir):
        threading.Thread.__init__(self)
        self.indir = indir
        self.outdir = outdir
        self.idrem_dir = idrem_dir
    def run(self):
        
        command = 'cd '+ str(self.idrem_dir)+' && java -Xmx8G -jar idrem.jar -b %s %s'%(self.indir, self.outdir)
        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        print(p.stdout.read())
def run_certrain_Idrem(path, file_name, idrem_dir, species='Human', Minimum_Standard_Deviation = 0.01,Convergence_Likelihood=0.1,Minimum_Absolute_Log_Ratio_Expression=0.05, trained=False):
    '''
    train IDREM model and save the results in iterative training with midpath and iteration
    
    parameters
    ----------------------
    paths: the path of IPF progression
    idremInput: average gene expression of each path
    trained: if the model is trained, use saved model
    
    '''
    path = Path(path)
    prefix_filename = file_name.split('.')[0]
    print('?')
    if os.path.exists(path / file_name):
        shutil.rmtree(path / file_name)
    
    # fidn the parent directory of path
    midpath = path.parent
    if os.path.exists(midpath / 'idremInput' / file_name):
        os.remove(midpath / 'idremsetting' / f'{prefix_filename}.txt')
    if os.path.exists(midpath / 'idremModel' / file_name):
        os.remove(midpath / 'idremModel' / f'{prefix_filename}.txt')

    dir1 = midpath / 'idremInput'
    dir2 = midpath / 'idremsetting'
    dir3 = midpath / 'idremModel'



    examplefile_path = idrem_dir / 'example_settings.txt'
    settings_file_path = midpath / 'idremsetting' / f'{prefix_filename}.txt'

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
                    elif species =='Mouse':
                        f.write('%s\t%s\n' % ('TF-gene_Interaction_Source', 'mouse_predicted.txt.gz'))
                        continue
                elif k == 2:
                    if species == 'Human':
                        f.write('%s\t%s\n' % ('TF-gene_Interactions_File', 'TFInput/human_encode.txt.gz'))
                        continue
                    elif species == 'Mouse':
                        f.write('%s\t%s\n' % ('TF-gene_Interactions_File', 'TFInput/mouse_predicted.txt.gz'))
                        continue
                elif k == 5:
                    if species == 'Human':
                        f.write('%s\t%s\n' % ('Gene_Annotation_Source', 'Human (EBI)'))
                        continue
                    elif species == 'Mouse':
                        f.write('%s\t%s\n' % ('Gene_Annotation_Source', 'Mouse (EBI)'))
                        continue
                elif k == 6:
                    if species == 'Human':
                        f.write('%s\t%s\n' % ('Gene_Annotation_File', 'goa_human.gaf.gz'))
                        continue 
                    elif species == 'Mouse':
                        f.write('%s\t%s\n' % ('Gene_Annotation_File', 'goa_mouse.gaf.gz'))
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
                    f.write('%s\t%s\n' % ('Expression_Data_File', os.path.join(os.path.abspath(os.path.join(midpath, 'idremInput')), f'{prefix_filename}.txt')))

    
    
    indir = settings_file_path
    outdir = midpath / 'idremModel' / f'{prefix_filename}.txt'

    threads = []

    threads.append(IDREMthread(indir, outdir,idrem_dir))
    for each in threads:
        each.start()
    for each in threads:
        each.join()

    dir1 = os.path.join(midpath, 'idremResults', file_name)
    dir2 = os.path.join(midpath, 'idremInput',file_name)
    command = ['mv '+dir2+ ' '+dir1]

    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    print(p.stdout.read())
    

if __name__ == "__main__":
    import  argparse
    import os 
    parser = argparse.ArgumentParser(description='Test the branches in the IDREM json file')
    parser.add_argument('--path', type=str, help='the file path of IDREM results')
    args = parser.parse_args()
    idrem_dir = '/mnt/md0/yumin/UNAGI/idrem'
    file_names = os.listdir(args.path)
    for file_name in file_names:
        print(file_name)
        if os.path.isdir(os.path.join(args.path, file_name)):
            print(file_name)

            count = 0
            candidates = [[0.05,0.1,0.04],[0.05,0.1,0.03],[0.05,0.1,0.02],[0.01,0.1,0.04],[0.01,0.1,0.03],[0.01,0.05,0.03],[0.01,0.05,0.02],[0.01,0.1,0.02],[0.01,0.05,0.01],[0.1,0.05,0.2],[0.2,0.1,0.3]]

            while not False:
            # # while not test_idrem_results(args.path, file_name):
                print(count)
                if count >= len(candidates):
                    print('all candidates parameters are tested')
                    break
                run_certrain_Idrem(args.path, file_name, idrem_dir, species='Human', Minimum_Standard_Deviation = candidates[count][0],Convergence_Likelihood=candidates[count][1],Minimum_Absolute_Log_Ratio_Expression=candidates[count][2],trained=False)
                count += 1
                
