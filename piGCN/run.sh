CUDA_VISIBLE_DEVICE=2

python sgc.py --dataset cora --lr 0.2 --weight 5e-3 --khop 1 --cuda
python sgc.py --dataset cora --lr 0.2 --weight 5e-3 --khop 2 --cuda
python sgc.py --dataset cora --lr 0.2 --weight 5e-3 --khop 1 --pinv
