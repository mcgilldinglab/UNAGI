import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

_IDREM_JSON_CACHE = {}


def _default_idrem_dir_from_repo() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    third_party = repo_root / "third_party" / "idrem"
    if third_party.exists():
        return third_party
    return repo_root / "idrem"


def _idrem_json_fingerprint(path):
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return f"{resolved}:{stat.st_size}:{int(stat.st_mtime_ns)}"


def clear_idrem_json_cache():
    _IDREM_JSON_CACHE.clear()


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
    json_path = Path(path).expanduser().resolve() / filename / 'DREM.json'
    fingerprint = _idrem_json_fingerprint(json_path)
    cached = _IDREM_JSON_CACHE.get(fingerprint)
    if cached is not None:
        return cached

    started_at = time.time()
    with json_path.open("r", encoding="utf-8") as handle:
        payload = "".join(handle.readlines())
    payload = payload[5:-2] + ']'
    parsed = json.loads(payload, strict=False)
    _IDREM_JSON_CACHE[fingerprint] = parsed
    elapsed = time.time() - started_at
    print(
        f"[IDREM] parse file={filename} cache=miss "
        f"elapsed={elapsed:.2f}s"
    )
    return parsed

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
        result = subprocess.run(
            ['java', '-Xmx8G', '-jar', 'idrem.jar', '-b', str(self.indir), str(self.outdir)],
            cwd=str(self.idrem_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if result.stdout:
            print(result.stdout)
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

    dir1 = midpath / 'idremResults' / file_name
    dir2 = midpath / 'idremInput' / file_name
    shutil.move(str(dir2), str(dir1))
    

if __name__ == "__main__":
    import  argparse
    import os 
    parser = argparse.ArgumentParser(description='Test the branches in the IDREM json file')
    parser.add_argument('--path', type=str, help='the file path of IDREM results')
    args = parser.parse_args()
    default_idrem_dir = _default_idrem_dir_from_repo()
    if not default_idrem_dir.exists():
        default_idrem_dir = Path(args.path).expanduser().resolve().parent / 'idrem'
    idrem_dir = Path(os.environ.get('UNAGI_IDREM_DIR', str(default_idrem_dir))).expanduser()
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
                
