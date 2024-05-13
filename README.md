# Cleaner for Stable Diffusion WebUI
从 `https://github.com/novitalabs/sd-webui-cleaner`项目fork出来的，根据需要增加了识别并去除图片文字的功能

## Installation

Clone this project in the WEBUI extensions folder(stable-diffusion-webui/extensions目录)

```
git clone https://github.com/yanming3/sd-webui-cleaner
```
<br>

注意：该项目依赖litelama和easyocr,如果stable-diffusion-webui禁用了自动安装功能，请手动安装:

```
pip install litelama==0.1.7
pip install easyocr==1.7.1
```

### API

```
//request-----------------------------------
POST http://127.0.0.1:7860/cleanup

body:
{
    "input_image": "<image base64 string>"
}


//response-----------------------------------
{
  "code": 0,  // 0:success
  "message": "ok",
  "image": "<image base64 string>"
}
```

<br>

### Used without GPU
If you don't have a GPU, please set the cleaner_use_cpu parameter to true through the setting page or api.

<br>

## Thanks
- https://github.com/advimman/lama
- https://github.com/Sanster/lama-cleaner
- https://github.com/novitalabs/sd-webui-cleaner
