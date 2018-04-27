# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import common
import seaborn as sns

setname = 'DENIS'
size    = (256, 256)if setname in ['SDO', 'SDO-Hard'] else None
fformat = '.jpg' if setname in ['CELEBa', 'INRIA'] else '.png'

training = common.load_images(f'../Clean/sets/{setname}-training', fformat)
testing  = common.load_images(f'../Clean/sets/{setname}-testing', fformat)

#%%

mses = np.zeros(len(testing))

for i, x in enumerate(testing):
    err = np.mean(np.mean((training - x)**2,1),1)
    mses[i] = err.min()
    
#%%
plt.figure()
plt.title(f'{setname} : {np.mean(mses):0.2e}')
sns.distplot(mses)
plt.xlabel('Closest Error')
plt.ylabel('Frequency')    
plt.grid()
plt.savefig(f'alignement_{setname}.eps')
np.savetxt(f"{setname}_align.csv", mses)