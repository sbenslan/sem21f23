from collections import namedtuple
import os
import json
from functools import partial

import shutil

from experiments.logbook import __ALIGN_CV_FOLDS__


ExperimentDesc = namedtuple('ExperimentDesc', ['path', 'topology', 'source', 'ckpt'])


def discover_topology(path):
    with open(os.path.join(path, 'config.json'), 'r') as fp:
        topology = json.load(fp)['network']['class']
    return topology


def find_topology_definition(quantlab_home, problem, topology):
    source_file = topology.lower() + '.py'
    return os.path.join(quantlab_home, 'problems', problem, topology, source_file)


def find_best_checkpoint(i_fold, path):

    def _fold_folder_name(i_fold):
        return 'fold'+str(i_fold).rjust(__ALIGN_CV_FOLDS__, '0')

    fold_path = os.path.join(path, _fold_folder_name(i_fold))
    return os.path.join(fold_path, 'saves', 'best.ckpt')


def filter_topology(topology, experiment_desc):
    return experiment_desc.topology == topology


problem = 'ImageNet'
topology = None
i_fold = 0

quantlab_home = os.path.join(os.getenv('HOME'), 'MSDocuments', 'QuantLab')
experiments_directory = os.path.join(quantlab_home, 'problems', problem, 'logs')
experiments_paths = [os.path.join(experiments_directory, d.name) for d in os.scandir(experiments_directory) if d.is_dir()]

topologies = list(map(discover_topology, experiments_paths))
sources = list(map(partial(find_topology_definition, quantlab_home, problem), topologies))
ckpts = list(map(partial(find_best_checkpoint, i_fold), experiments_paths))

experiments_descs = list()
for path, topology, source, ckpt in zip(experiments_paths, topologies, sources, ckpts):
    experiments_descs.append(ExperimentDesc(path, topology, source, ckpt))

if topology:
    # retrieve only experiments for the given topology
    experiments_list = list(filter(partial(filter_topology, topology), experiments_descs))


def create_converter_reqs(experiment_desc, target_folder):

    try:
        os.makedirs(target_folder, exist_ok=False)

        source_filename = os.path.basename(experiment_desc.source)
        shutil.copy(experiment_desc.source, os.path.join(target_folder, source_filename))

        ckpt_filename = os.path.basename(experiment_desc.source)
        shutil.copy(experiment_desc.ckpt, os.path.join(target_folder, ckpt_filename))

    except OSError:

        print("It seems that the desired experiment has already been exported.")
        return
