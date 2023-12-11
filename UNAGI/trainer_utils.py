import numpy as np
from scipy.stats import rankdata

def transfer_to_ranking_score(gw):
    '''
    ranking score
    '''
    # gw = adata.layers['geneWeight'].toarray()
    od = gw.shape[1]-rankdata(gw,axis=1)+1
    score = 1+1/np.power(od,0.5)
    
    return score