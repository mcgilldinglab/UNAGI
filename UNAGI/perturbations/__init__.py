'''
The perturbations module allows to perform the pathway and compounds in-silico perturbations and obatin the perturbation results.
'''

from .compounds import get_top_compounds, get_top_single_genes, get_top_gene_combinations
from .pathways import get_top_pathways