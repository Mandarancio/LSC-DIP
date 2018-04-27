# -*- coding: utf-8 -*-
"""
Conf loader for LSC and DIP project.

@author: Martino Ferrari
@email: manda.mgf@gmail.com
"""
import yaml
import common


class Conf(object):
    """LSC Configuration class."""
    def __init__(self, conf):
        self.__dict__.update(conf)
        self.conf = conf
        
    def size(self):
        return  None if self.resize == 'none' else self.resize

    def trainingset_path(self):
        return f'{self.datafolder}/{self.dataset}-training/'
    
    def testingset_path(self):
        return f'{self.datafolder}/{self.dataset}-testing/'
    
    def codebook_name(self):
        return f'{self.dataset}_{self.resize}_{self.vectorization}_{self.training["sort"]}s_{self.nbands}b_{self.training["n_codes"]}c_{self.training["batch"]}bs_{self.training["n_levels"]}l.pkl'

    def vect_functions(self):
        # default (np.reshape(-1)), zigzag, spiral and circular
        if self.vectorization == 'zigzag':
            return common.zigzag_vect, common.zigzag_mat
        if self.vectorization == 'spiral':
            return common.sim_spiral_vect, common.sim_spiral_mat
        if self.vectorization == 'circular':
            return common.sim_circ_vect, common.sim_circ_mat
        if self.vectorization == 'default':
            return common.default_vect, common.default_mat
        raise ValueError(f'''"{self.vectorization}" not recognized, only options are:
            - default
            - zigzag
            - spiral
            - circular''')

    def save_cfg(self, folder):
        with open(folder+'/config.yml', 'w') as f:
            yaml.dump(self.conf, f)

            
def load_conf(path):
    """load configuration."""
    with open(path, 'r') as f:
        adict = yaml.load(f)
        return Conf(adict)
    raise ValueError(f'Configuration not found at: {path}.')
    