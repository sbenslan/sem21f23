from collections import namedtuple
import os
import json
from functools import partial

import torch
import shutil

from experiments.logbook import __ALIGN_CV_FOLDS__


ExperimentDesc = namedtuple('ExperimentDesc', ['dir', 'config', 'topology', 'source', 'ckpt'])

# VOCABULARY:
#
# filesystem := collection of nodes and arcs; a graph G = (N, E)
#               for simplicity, we consider only two kinds of nodes: files and directories
# path       := a (cycle-free) sequence of nodes {N_{0}, N_{1}, N_{2}, ..., N_{k}}
# directory  := a non-file node
# file       := a non-directory node
#
# I identify directories and files with the following rule.
# Assign a name N.name to each node N.
# Let P be a path.
# I assume that each path starts with the node N_{0} which has the name N_{0}.name := '/'.
# Suppose the terminal node N_{k} is a directory; the definition will be analogous for files.
# We say that "the path points to a directory".
# We identify the directory with the juxtaposition of the strings representing the nodes' names:
#
#       directory == '/' + 'N_{1}.name' + '/' + 'N_{2}.name' + ... + '/' + 'N_{k}.name'  (*)
#
# Given a directory (*) and a node 'N_{k+1}', I say that I can "extend" the path by juxtaposing '/N_{k+1}.name' to the directory.


def find_configuration(dir_exp):
    files = [f.name for f in os.scandir(dir_exp) if f.is_file()]
    if 'config.json' in files:
        return 'config.json'
    else:
        print("No configuration was found at {}".format(dir_exp))
        return None


def discover_topology(file_config):
    with open(file_config, 'r') as fp:
        topology = json.load(fp)['network']['class']
    return topology


def find_topology_definition(dir_problem_home, topology):
    source_file = topology.lower() + '.py'
    return os.path.join(dir_problem_home, topology, source_file)


def find_best_checkpoint(i_fold, dir_exp):

    def _fold_folder_name(i_fold):
        return 'fold'+str(i_fold).rjust(__ALIGN_CV_FOLDS__, '0')

    return os.path.join(_fold_folder_name(i_fold), 'saves', 'best.ckpt')


def filter_topology(topology, experiment_desc):
    return experiment_desc.topology == topology


problem = 'ImageNet'
topology = None
i_fold = 0

dir_quantlab_home = os.path.join(os.getenv('HOME'), 'MSDocuments', 'QuantLab')
dir_problem_home  = os.path.join(dir_quantlab_home, 'problems', problem)
dir_logs          = os.path.join(dir_problem_home, 'logs')
dirs_experiments  = [os.path.join(dir_logs, d.name) for d in os.scandir(dir_logs) if d.is_dir()]

configs    = list(map(find_configuration, dirs_experiments))
topologies = list(map(discover_topology, [os.path.join(dir_exp, cfg_file) for dir_exp, cfg_file in zip(dirs_experiments, configs)]))
t_sources  = list(map(partial(find_topology_definition, dir_problem_home), topologies))
ckpts      = list(map(partial(find_best_checkpoint, i_fold), dirs_experiments))

experiments_descs = [ExperimentDesc(dir, config, topology, source, ckpt) for dir, config, topology, source, ckpt in zip(dirs_experiments, configs, topologies, t_sources, ckpts)]
experiments_descs = sorted(experiments_descs, key=lambda desc: desc.dir)

if topology:
    # retrieve only experiments for the given topology
    experiments_list = list(filter(partial(filter_topology, topology), experiments_descs))


def export_converter_reqs(experiment_desc, dir_target):
    """
    Export the information needed by the PyTorch-2-Caffe conversion tool.
    """
    def _python_touch():
        init_file = os.path.join(dir_target, '__init__.py')
        with open(init_file, 'w') as fp:
            os.utime(init_file, None)

    try:
        os.makedirs(dir_target, exist_ok=False)
        _python_touch()

        # copy topology configuration (whole experiment configuration is not needed)
        file_config_source = os.path.join(experiment_desc.dir, experiment_desc.config)
        file_config_target = os.path.join(dir_target, experiment_desc.config)
        with open(file_config_source, 'r') as fp_src, open(file_config_target, 'w') as fp_tgt:
            net_config_src = json.load(fp_src)['network']
            net_config_tgt = {
                'class':    net_config_src['class'],
                'params':   net_config_src['params']
            }
            json.dump(net_config_tgt, fp_tgt, indent=4)

        # copy topology source file
        source_filename = os.path.basename(experiment_desc.source)
        shutil.copy(experiment_desc.source, os.path.join(dir_target, source_filename))

        # copy network state (whole experiment state is not needed)
        ckpt_dict = torch.load(os.path.join(experiment_desc.dir, experiment_desc.ckpt))
        ckpt_filename = os.path.basename(experiment_desc.ckpt)
        torch.save(ckpt_dict['network'], os.path.join(dir_target, ckpt_filename))

    except OSError:

        print("It seems that the desired experiment has already been exported.")
        return
