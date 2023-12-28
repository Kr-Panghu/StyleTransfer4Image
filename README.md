<h1 align="center">
Style Transfer for Image
</h1>

<p align="center">Source code for StyleTransfer4Image, where we attempt to do multiple applications based on diffusion models.</p>

<p align="center">Collaborators: 
<a href="https://github.com/yangyuxiao-sjtu">YuxiaoYang</a> 
<a href="https://github.com/ZhangTian-Yu">TianyuZhang</a>
<a href="https://github.com/ParkCorsa">ZiqiHuang</a>
<a href="https://github.com/shuzechen">ShuzeChen</a>
<a href="https://github.com/Kr-Panghu">KrCen</a>
</p>

![](./doc/title.jpg)

## QuickStart

Clone Repo:

~~~
git clone https://github.com/Kr-Panghu/StyleTransfer4Image
~~~

Install submodules dependencies:

~~~
cd CLIP-Based-StyleTransfer
pip install -e ./CLIP & pip install -e ./guided_diffusion
~~~

Download pre-trained checkpoints:

~~~
wget -O 256x256_diffusion_uncond.pt https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
~~~

For checkpoint trained by ourselves, download from [google drive](https://drive.google.com/file/d/1i4dhiyLCR9z7uN4OU6Rwuj8-PfDNM8VU/view?usp=drive_link) and put it into `CLIP-Based-StyleTransfer/models/` for sampling.

## Generate Samples with Diffusion

Here we trained [improved-diffusion](https://github.com/openai/improved-diffusion) models on 22000 pictures from [ffhq_dataset_thumbnails128x128](https://github.com/NVlabs/ffhq-dataset). By leveraging diffusion model, we are able to sample mutiple generated faces with good quality.

The following figures are some random-picked samples.

![result0](./doc/result0.png)

If you want to make your own sample generated faces from random gaussian noise:

~~~
cd ./CLIP-Based-StyleTransfer/guided_diffusion

MODEL_FLAGS="--image_size 64 --num_channels 64 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 64"

python3 guided_diffusion/scripts/image_sample.py --model_path models/ema_0.9999.pt $MODEL_FLAGS $DIFFUSION_FLAGS
~~~

---

## CLIP-Based Style Transfer

> Implementation of Zero-Shot Contrastive Loss for Text-Guided Diffusion Image Style Transfer.

### Zero-Shot Contrastive Loss for Text-Guided Diffusion Image Style Transfer (ICCV 2023 Accepted)

[![arXiv](https://img.shields.io/badge/arXiv-2303.08622-b31b1b.svg)](https://arxiv.org/abs/2303.08622) 
#### [Paper](https://arxiv.org/pdf/2303.08622.pdf)&ensp;&ensp;&ensp; [Project Page](https://github.com/YSerin/ZeCon)


In these images, we show the results of generating multiple styles using pre-trained [CLIP](https://github.com/openai/CLIP) models and diffusion models. These styles include different facial expressions, artistic styles, etc. By leveraging diffusion model, we are able to generate multiple different styles in one run without the need to re-train the model or change parameters.

![result1](./doc/result1.png)

Additionally, we are able to control facial expression.

![result2](doc/result2.png)

If you want to do text-guided image style transfer, one example is:

~~~
python3 style_transfer.py --path_to_image './photo.png' --output_dir './results' --source 'Photo' --target 'Ukiyo-e' --model '256x256_diffusion_uncond.pt'
~~~

The `--source` is the initial prompt of input image. `--target` is the control signal for generation. `--model` gives the pre-trained weights of the model, you should use the checkpoint downloaded from OpenAI. The code was tested on a RTX4090ti 24GB but should work on other cards with at least 15GB VRAM.

---

## Image-2-Image Style Transfer

To be determined.

---

## TBD

TBD