import scanpy as sc
import argparse
import numpy as np
from pathlib import Path
def parse_gmt(file_path):
    """
    Parses a GMT file and returns a dictionary of gene sets.
    
    -------------
    Parameters:
        file_path (str): Path to the GMT file.

    Returns:
        dict: A dictionary where keys are gene set names and values are lists of genes.
    """
    gene_sets = {}
    file_path = Path(file_path)
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line by tabs
            parts = line.strip().split('\t')
            
            # The first column is the gene set name
            gene_set_name = parts[0]
            
            # The second column is the description (optional, can be ignored)
            description = parts[1]
            
            # The remaining columns are the genes
            genes = parts[2:]
            
            # Store the gene set in the dictionary
            gene_sets[gene_set_name] = genes

    return gene_sets

def find_overlapping_genes(gene_names, gene_sets):
    """
    Finds overlapping genes between a list of gene names and a dictionary of gene sets.

    -------------
    Parameters:
        gene_names (list): A list of gene names.
        gene_sets (dict): A dictionary where keys are gene set names and values are lists of genes.

    Returns:
        dict: A dictionary where keys are gene set names and values are lists of overlapping genes.
    """
    overlapping_genes = {}
    hits = 0
    for gene_set_name, genes in gene_sets.items():
        # Find the intersection of the two sets
        overlap = set(gene_names).intersection(set(genes))
        
        if overlap:
            overlapping_genes[gene_set_name] = list(overlap)
            hits += len(overlap)
    if hits < 1:
        raise ValueError("No overlapping genes found. Please check the input file or gene names.")
    return overlapping_genes
    
def preprocess_msigdb(db_directory,data,output):
    '''
    Preprocess the MSigDB GMT file and find overlapping genes with the input data.
    
    -------------
    Parameters:
    db_directory : str
        Path to the GMT file.
    data : str
        Path to the input data file.
    output : str
        Path to save the output file.'''
    adata = sc.read_h5ad(data)
    if adata.var_names is None:
        raise ValueError("adata.var_names is None. Please check the input file.")
    gene_names = adata.var_names
    print('processing your gene sets database....')
    parsed_file = parse_gmt(db_directory)
    
    processed_pathway_db = find_overlapping_genes(gene_names, parsed_file)
    lens = []
    for each in processed_pathway_db:
        lens.append(len(processed_pathway_db[each]))
    print('average number of genes in each gene set:', np.mean(lens))
    print('max number of genes in each gene set:', np.max(lens))
    print('min number of genes in each gene set:', np.min(lens))
    print('median number of genes in each gene set:', np.median(lens))
    np.save(output, processed_pathway_db)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a GMT file and find overlapping genes.")
    parser.add_argument("--db_directory", type=str, required=True, help="Path to the GMT file.")
    parser.add_argument("--data", type=str, required=True, help="Path to the input data file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output file.")
    
    args = parser.parse_args()
    
    preprocess_msigdb(args.db_directory,args.data,args.output)