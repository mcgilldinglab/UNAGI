import os
import shutil
import warnings
from UNAGI import UNAGI


DATA_DIR = '/datasets/dcfm_project/ipf_data_copy_01'
FOLDERS_TO_REMOVE = ['0', 'model_save']


def remove_folders(path: str, folders: list):
    """
    Remove specified folders under a given path.
    
    Args:
        path (str): The parent directory path.
        folders (list): List of folder names to remove.
    """
    for folder in folders:
        folder_path = os.path.join(path, folder)
        try:
            shutil.rmtree(folder_path)
            print(f"Removed folder: {folder_path}")
        except FileNotFoundError:
            print(f"Folder not found, skipping: {folder_path}")
        except Exception as e:
            print(f"Failed to remove {folder_path}: {e}")


def main():
    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Initialize UNAGI
    unagi = UNAGI()

    # --- Step 1: Clean up folders ---
    remove_folders(DATA_DIR, FOLDERS_TO_REMOVE)

    # --- Step 2: Verify data directory ---
    try:
        contents = os.listdir(DATA_DIR)
        print(f"Found {len(contents)} items in {DATA_DIR}")
    except Exception as e:
        raise RuntimeError(f"Failed to list contents of {DATA_DIR}: {e}")

    # --- Step 3: Run UNAGI pipeline ---
    unagi.setup_data(
        data_path=DATA_DIR,
        total_stage=4,
        stage_key='stage'
    )

    unagi.setup_training(
        task="IPF_main",
        dist="ziln",          # paper’s choice for IPF
        device="cuda:2",      # use your GPU; change if needed
        epoch_iter=10,        # default used by the package unless overridden
        epoch_initial=20,     # default used by the package unless overridden
        lr=1e-4,              # package default
        lr_dis=5e-4,          # package default (discriminator)
        beta=1.0,             # package default
        hidden_dim=256,       # package default
        latent_dim=64,        # package default
        graph_dim=1024,       # package default
        BATCHSIZE=512,        # package default
        max_iter=10,          # outer iterations
        GPU=True,             # run on CUDA
        adversarial=True,     # paper uses GAN
        GCN=True              # paper uses GCN
    )

    unagi.run_UNAGI(idrem_dir='../idrem')


if __name__ == "__main__":
    main()
