import os
import shutil
import subprocess
import threading
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..utils.idrem_helper import run_certrain_Idrem, test_idrem_results


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

    for each in edges[0]:
        if str(each[0]) not in paths:
            paths[str(each[0])] = [[each[0]], [each[1]]]
        else:
            paths[str(each[0])][1].append(each[1])

    for stage in range(1, total_stages - 1):
        for each in edges[stage]:
            for item in paths.keys():
                if len(paths[item]) <= stage:
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
        if len(path) == total_stages:
            stages = [averageNode(node, state[i]) for i, node in enumerate(path)]
            reshaped_stages = [stage.reshape(-1, 1) for stage in stages]
            joint_matrix = np.concatenate(reshaped_stages, axis=1)
            out.append(joint_matrix)

    return out


def getIdrem(paths, state):
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
    for each in paths:
        joint_matrix = np.concatenate(
            (
                state[0][each[0]].reshape(-1, 1),
                state[1][each[1]].reshape(-1, 1),
                state[2][each[2]].reshape(-1, 1),
                state[3][each[3]].reshape(-1, 1),
            ),
            axis=1,
        )
        out.append(joint_matrix)
    return out


class IDREMthread(threading.Thread):
    '''the thread for running IDREM. Support multiple threads.'''

    def __init__(self, indir, outdir, each, idrem_dir):
        threading.Thread.__init__(self)
        self.indir = indir
        self.outdir = outdir
        self.each = each
        self.idrem_dir = idrem_dir

    def run(self):
        command = "cd " + str(self.idrem_dir) + " && java -Xmx8G -jar idrem.jar -b %s %s" % (self.indir, self.outdir)
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True, check=False)


def _available_memory_gb():
    try:
        import psutil
        return int(psutil.virtual_memory().available // (1024 ** 3))
    except Exception:
        if hasattr(os, 'sysconf') and 'SC_AVPHYS_PAGES' in os.sysconf_names and 'SC_PAGE_SIZE' in os.sysconf_names:
            pages = os.sysconf('SC_AVPHYS_PAGES')
            page_size = os.sysconf('SC_PAGE_SIZE')
            return int((pages * page_size) // (1024 ** 3))
    return 1


def _default_idrem_worker_cap(job_count):
    available_ram_gb = _available_memory_gb()
    cpu_cap = max(1, os.cpu_count() or 1)
    memory_cap = max(1, available_ram_gb // 10)
    return max(1, min(job_count, 4, cpu_cap, memory_cap))


def _run_idrem_job(indir, outdir, idrem_dir):
    command = "cd " + str(idrem_dir) + " && java -Xmx8G -jar idrem.jar -b %s %s" % (indir, outdir)
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True, check=False)


def runIdrem(
    paths,
    midpath,
    idremInput,
    genenames,
    iteration,
    idrem_dir,
    species='Human',
    Minimum_Standard_Deviation=0.01,
    Convergence_Likelihood=0.1,
    Minimum_Absolute_Log_Ratio_Expression=0.05,
    trained=False,
):
    '''train IDREM model and save the results in iterative training with midpath and iteration.'''
    run_started_at = time.time()
    midpath = Path(midpath)
    idrem_dir = Path(idrem_dir)
    dir1 = midpath / str(iteration) / 'idremInput'
    dir2 = midpath / str(iteration) / 'idremsetting'
    dir3 = midpath / str(iteration) / 'idremModel'

    dir1.mkdir(parents=True, exist_ok=True)
    dir2.mkdir(parents=True, exist_ok=True)
    dir3.mkdir(parents=True, exist_ok=True)

    if len(paths) == 0:
        raise ValueError("No paths found in the input data.")

    print(f"[IDREM] iteration {iteration}: preparing {len(paths)} trajectory settings")
    settings_started_at = time.time()

    for i, each in enumerate(paths):
        each_processed = []
        for e in each:
            e = str(e).strip('[]')
            e = e.replace(', ', 'n')
            each_processed.append(e)

        file_name = '-'.join(each_processed)
        file_path = midpath / str(iteration) / 'idremInput' / f'{file_name}.txt'
        header = ['gene'] + [f'stage{j}' for j in range(len(each))]
        with open(file_path, 'w') as f:
            f.write('\t'.join(header) + '\n')
            for j, row in enumerate(idremInput[i]):
                row_data = '\t'.join(str(r) for r in row)
                f.write("%s\t%s\n" % (genenames[j], row_data))

        examplefile_path = idrem_dir / 'example_settings.txt'
        settings_file_path = midpath / str(iteration) / 'idremsetting' / f'{file_name}.txt'
        with open(examplefile_path, 'r') as examplefile:
            with open(settings_file_path, 'w') as f:
                for k, line in enumerate(examplefile.readlines()):
                    if trained and k == 4:
                        f.write(
                            '%s\t%s\n'
                            % (
                                'Saved_Model_File',
                                os.path.join(os.path.abspath(os.path.join(midpath, str(iteration), 'idremInput')), f'{file_name}.txt'),
                            )
                        )
                    elif k == 1:
                        if species == 'Human':
                            f.write('%s\t%s\n' % ('TF-gene_Interaction_Source', 'human_encode.txt.gz'))
                            continue
                        if species == 'Mouse':
                            f.write('%s\t%s\n' % ('TF-gene_Interaction_Source', 'mouse_predicted.txt.gz'))
                            continue
                    elif k == 2:
                        if species == 'Human':
                            f.write('%s\t%s\n' % ('TF-gene_Interactions_File', str(idrem_dir.resolve() / 'TFInput/human_encode.txt.gz')))
                            continue
                        if species == 'Mouse':
                            f.write('%s\t%s\n' % ('TF-gene_Interactions_File', str(idrem_dir.resolve() / 'TFInput/mouse_predicted.txt.gz')))
                            continue
                    elif k == 5:
                        if species == 'Human':
                            f.write('%s\t%s\n' % ('Gene_Annotation_Source', 'Human (EBI)'))
                            continue
                        if species == 'Mouse':
                            f.write('%s\t%s\n' % ('Gene_Annotation_Source', 'Mouse (EBI)'))
                            continue
                    elif k == 6:
                        if species == 'Human':
                            f.write('%s\t%s\n' % ('Gene_Annotation_File', str(idrem_dir.resolve() / 'goa_human.gaf.gz')))
                            continue
                        if species == 'Mouse':
                            f.write('%s\t%s\n' % ('Gene_Annotation_File', str(idrem_dir.resolve() / 'goa_mouse.gaf.gz')))
                            continue
                    elif k == 17:
                        f.write('%s\n' % ('miRNA-gene_Interaction_Source'))
                        continue
                    elif k == 18:
                        f.write('%s\n' % ('miRNA_Expression_Data_File'))
                        continue
                    elif k == 26:
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
                    elif k == 51:
                        f.write('%s\t%s\n' % ('Convergence_Likelihood_%', str(Convergence_Likelihood)))
                        continue
                    elif k == 52:
                        f.write('%s\t%s\n' % ('Minimum_Standard_Deviation', str(Minimum_Standard_Deviation)))
                        continue
                    elif k != 3:
                        f.write(line)
                    else:
                        f.write(
                            '%s\t%s\n'
                            % (
                                'Expression_Data_File',
                                os.path.join(os.path.abspath(os.path.join(midpath, str(iteration), 'idremInput')), f'{file_name}.txt'),
                            )
                        )

    settings_elapsed = time.time() - settings_started_at
    print(
        f"[IDREM] iteration {iteration}: settings_generation "
        f"elapsed={settings_elapsed:.2f}s"
    )

    settinglist = sorted([each for each in os.listdir(midpath / str(iteration) / 'idremsetting') if each and each[0] != '.'])
    print(f"[IDREM] iteration {iteration}: running {len(settinglist)} iDREM jobs")

    execution_started_at = time.time()
    jobs = []
    for each in settinglist:
        indir = os.path.abspath(midpath / str(iteration) / 'idremsetting' / each)
        outdir = os.path.join(os.path.abspath(midpath / str(iteration) / 'idremModel'), each)
        jobs.append((indir, outdir))

    worker_cap = _default_idrem_worker_cap(len(jobs)) if jobs else 1
    if jobs:
        with ThreadPoolExecutor(max_workers=worker_cap) as executor:
            futures = [
                executor.submit(_run_idrem_job, indir, outdir, idrem_dir)
                for indir, outdir in jobs
            ]
            for future in as_completed(futures):
                future.result()
    execution_elapsed = time.time() - execution_started_at
    print(
        f"[IDREM] iteration {iteration}: external_run "
        f"elapsed={execution_elapsed:.2f}s worker_cap={worker_cap}"
    )

    collection_started_at = time.time()
    if not trained:
        results_dir = midpath / str(iteration) / 'idremResults'
        if results_dir.exists():
            shutil.rmtree(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        idrem_input_dir = midpath / str(iteration) / 'idremInput'
        for viz_file in idrem_input_dir.glob('*.txt_viz'):
            shutil.move(str(viz_file), str(results_dir / viz_file.name))
    else:
        results_dir = midpath / str(iteration) / 'idremResults'

    idrem_inputs = sorted(os.listdir(midpath / str(iteration) / 'idremInput'))
    collection_elapsed = time.time() - collection_started_at
    validation_started_at = time.time()
    for each_input in idrem_inputs:
        each_input = each_input + '_viz'
        count = 0
        candidates = [
            [0.05, 0.1, 0.04],
            [0.05, 0.1, 0.03],
            [0.05, 0.1, 0.02],
            [0.01, 0.1, 0.04],
            [0.01, 0.1, 0.03],
            [0.01, 0.05, 0.03],
            [0.01, 0.05, 0.02],
            [0.01, 0.1, 0.02],
            [0.01, 0.05, 0.01],
            [0.1, 0.05, 0.2],
            [0.2, 0.1, 0.3],
        ]

        while not test_idrem_results(results_dir, each_input):
            if count >= len(candidates):
                print(f"[IDREM] warning: all fallback parameter sets were tested for {each_input}")
                break
            run_certrain_Idrem(
                results_dir,
                each_input,
                idrem_dir,
                species=species,
                Minimum_Standard_Deviation=candidates[count][0],
                Convergence_Likelihood=candidates[count][1],
                Minimum_Absolute_Log_Ratio_Expression=candidates[count][2],
                trained=False,
            )
            count += 1
    validation_elapsed = time.time() - validation_started_at
    timings = {
        'iteration': int(iteration),
        'settings_generation_seconds': settings_elapsed,
        'external_run_seconds': execution_elapsed,
        'result_collection_seconds': collection_elapsed,
        'validation_seconds': validation_elapsed,
        'total_seconds': time.time() - run_started_at,
        'trajectory_count': int(len(paths)),
        'job_count': int(len(settinglist)),
        'worker_cap': int(worker_cap),
    }
    timing_path = midpath / str(iteration) / 'idrem_timing.json'
    timing_path.write_text(json.dumps(timings, indent=2), encoding='utf-8')
    print(
        f"[IDREM] iteration {iteration}: complete "
        f"collection_elapsed={collection_elapsed:.2f}s "
        f"validation_elapsed={validation_elapsed:.2f}s "
        f"total_elapsed={timings['total_seconds']:.2f}s"
    )


def averageNode(nodes, state):
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
        out += state[each]
    return out / len(nodes)


if __name__ == '__main__':
    edges = [
        [[2, 0], [0, 1], [8, 3], [3, 4], [3, 5], [4, 6], [5, 7], [6, 8], [1, 9]],
        [[1, 0], [4, 1], [3, 2], [0, 3], [7, 5], [0, 6], [8, 7], [10, 10], [3, 11], [8, 12]],
        [[1, 0], [0, 1], [1, 2], [3, 3], [5, 4], [0, 5], [2, 6], [7, 8], [5, 9], [12, 10], [2, 12]],
    ]
    paths = getClusterPaths(edges, 4)
    example_average_values = Path(__file__).resolve().parents[1] / 'data' / 'example' / '2' / 'averageValues.npy'
    averageValues = np.load(example_average_values, allow_pickle=True)
    idrem = getClusterIdrem(paths, averageValues, 4)
    print(idrem)
