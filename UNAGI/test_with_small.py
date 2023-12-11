from UNAGI import UNAGI
unagi = UNAGI()
unagi.setup_data('/mnt/md0/yumin/to_upload/UNAGI/data/small/0.h5ad',total_stage=4,stage_key='stage',splited_dataset=True)
unagi.setup_training(task='small',dist='ziln',device='cuda:0',GPU=True,epoch_iter=15,epoch_initial=10,max_iter=3,BATCHSIZE=560)
unagi.run_UNAGI()