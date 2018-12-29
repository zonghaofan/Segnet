# SegNet 用作卫星堆料分割,数据集
https://github.com/zonghaofan/Segnet

## Environments
- Python 3.5.2
- Numpy==1.13.3
- Pillow==4.2.1
- tensorflow-gpu==1.3.0
- tensorflow-tensorboard==0.1.7


## Usage  
1. 下载论文数据集。
```
bash download.sh
```

2. Training
```
python train.py

4. Evaluation
```
python test.py \
  --resdir test_results/pred
```
