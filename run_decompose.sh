GPU_ID=1

CUDA_VISIBLE_DEVICES="${GPU_ID}" python main_multiseed.py --parent_data_dir "cat_sculpture/" --node v0 --test_name "v0" --GPU_ID "${GPU_ID}" --multiprocess 0
CUDA_VISIBLE_DEVICES="${GPU_ID}" python main_multiseed.py --parent_data_dir "cat_sculpture/" --node v2 --test_name "v2" --GPU_ID "${GPU_ID}" --multiprocess 0
CUDA_VISIBLE_DEVICES="${GPU_ID}" python main_multiseed.py --parent_data_dir "cat_sculpture/" --node v1 --test_name "v1" --GPU_ID "${GPU_ID}" --multiprocess 0