## Requirements
* Python 3.10, Pytorch 2.4.0, [xformers](https://github.com/facebookresearch/xformers) 0.0.27.post2
* More detail (See [environment.yaml](environment.yaml))
* A suitable [conda](https://conda.io/) environment named `dmapsr` can be created and activated with:

```
conda create -n dmapsr python=3.10
conda activate dmapsr
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[torch]"
pip install -r requirements.txt
```


## Inference
### :rocket: Fast testing 
```
python inference_dmapsr.py -i [image folder/image path] -o [result folder] --num_steps 1
```
1. **To deal with large images, e.g., 1k---->4k, we recommend adding the option** ``--chopping_size 256``.
2. Other options:
    + Specify the pre-downloaded [SD Turbo](https://huggingface.co/stabilityai/sd-turbo) Model: ``--sd_path``.
    + Specify the pre-downloaded noise predictor: ``--started_ckpt_path``.
    + The number of sampling steps: ``--num_steps``.
    + If your GPU memory is limited, please add the option ``--chopping_bs 1``.

### :whale: Now also available in Docker
```bash
docker compose up -d # Go to http://127.0.0.1:7860/
```

### :airplane: Reproducing our paper results
+ Synthetic dataset of ImageNet-Test: [Google Drive](https://drive.google.com/file/d/1PRGrujx3OFilgJ7I6nW7ETIR00wlAl2m/view?usp=sharing).

+ Real data for image super-resolution: [RealSRV3](https://github.com/csjcai/RealSR) | [RealSet80](testdata/RealSet80)

+ To reproduce the quantitative results on Imagenet-Test and RealSRV3, please add the color fixing options by ``--color_fix wavelet``.

## Training
### :turtle: Preparing stage
1. Download the finetuned LPIPS model from this [link](https://huggingface.co/OAOA/InvSR/resolve/main/vgg16_sdturbo_lpips.pth?download=true) and put it in the folder of "weights".
2. Prepare the [config](configs/sd-turbo-sr-ldis.yaml) file:
    + SD-Turbo path: configs.sd_pipe.params.cache_dir.
    + Training data path: data.train.params.data_source.
    + Validation data path: data.val.params.dir_path (low-quality image) and data.val.params.extra_dir_path (high-quality image).
    + Batchsize: configs.train.batch and configs.train.microbatch (total batchsize = microbatch * #GPUS * num_grad_accumulation)

### :dolphin: Begin training
```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 --nnodes=1 main.py --save_dir [Logging Folder] 
```

### :whale: Resume from interruption
```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 --nnodes=1 main.py --save_dir [Logging Folder] --resume save_dir/ckpts/model_xx.pth
```
