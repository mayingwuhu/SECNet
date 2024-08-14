# SECNet 
Semantic Element Consistency Learning. PyTorch evaluate code for SECNet. 

## Installation 
### Dependencies
我们的代码在Ubuntu上运行，您需要准备一些材料
1. 打开(Link)(这部分内容过大放在Goodle中，由于匿名提交，Coming Soon)将ALPRO_model.pt、SECNet_model.pt、bert-base-uncased.zip、vit_base_patch16_224.zip、ext.zip下载到根目录下，并解压相应zip文件
2. 将videos.pt下载放置到data/msrvtt_ret目录下

最后的目录树为：
```
- SECNet 
  - bert-base-uncased 
  - config_release  
  - data 
    - msrvtt 
      - txt
      - videos
  - ext
  - output
  - run_scripts
  - src
  - vit_base_patch16_224
  - ALPRO_model.pt
  - SECNet_model.pt
```

### Installation Steps
```
conda create -n SECNet python=3.7
pip install cmake==3.13.0 
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install horovod
pip install -r requirements.txt
```


## Training SECNet
```
conda activate SECNet 
cd run_scripts 
bash pt_SECNet.sh 
```

## Test on MSRVTT 
```
conda activate SECNet  
cd run_scripts 
bash inf_msrvtt_ret.sh     
```
你可以将`config_release/msrvtt_ret.json`中的`e2e_weights_path`改为`SECNet_model`以获得SECNet的测试结果               

