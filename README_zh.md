# Model Toolkit of Anti-Fake AI

**通过构建Docker镜像，将属于你自己的模型接入Anti-Fake AI**



语言:
- [English](README.md)
- [中文](README_zh.md)


当前基础模型支持情况：

| 模态  | 支持情况 |  备注  |
| :---: | :------: | :----: |
| 图像  | :hammer: | 进行中 |
| 音频  |   :x:    |   -    |
| 视频  |   :x:    |   -    |
| 文本  |   :x:    |   -    |

:hammer: ：正在开发中

:x: ：暂不支持

## :rocket: 快速开始 

在`example`中提供了模板文件。按照模板，下面的步骤提供了一种添加自定义 `图像检测` 的实现方式。

1. 安装依赖

使用 `pip` 安装 `afk_tooset_http` 包。包文件可以在本项目的release中找到。
```shell
pip install afk_toolset_http.whl
```

2. 注册模型

在 `model_impy.py` 实现`get_sever`函数。该函数返回`AfkServer`对象，并在运行期注入到模型服务中。

```python
# model_impl.py
import torchvision
from torchvision import transforms
from afk_toolset_http import AfkServer, AfkImageModel

def get_server():
    conv_model = torchvision.models.convnext_base(num_classes=2).to("cuda").eval()
    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    model = AfkImageModel(conv_model, transform)
    model.load_checkpoint("/path/to/checkpoint") # 加载你自己的模型
    server = AfkServer("test_server")
    server.register_model(model)
    return server
```

3. 运行模型


现在你已经准备好了你的模型，只需启动服务，就能通过http请求的方式访问模型。我们已经准备好了`entry.py`。

```python
# entry.py
from afk_toolset_http import AfkServer
from afk_toolset_http import DetectRequest, DetectResult
from typing import Annotated
from fastapi import FastAPI, Depends
from model_impl import get_server

app = FastAPI()

@app.post("/detect")
async def detect(
    request: DetectRequest, server: Annotated[AfkServer, Depends(get_server)]
) -> DetectResult:
    result = server.perform_detect_single(request)
    return result

```
安装并使用 `uvicorn` 运行模型服务：
```shell
pip install "uvicorn[standard]"
uvicorn entry:app --host 0.0.0.0 --port 80
```
现在你能够从`80`端口访问模型了！


### :pencil: 自定义模型 

我们已经为你准备了基础的图像模型封装类AfkImageModel，能够满足基础的使用。但是，如果你需要高度自定义的模型，一个`MyModel`类, 并实现`detect`方法和`load_checkpoint`方法。

```python
# my_conv_model.py
from afk_toolset_http import AfkImageModel
from torchvision import transforms, models
import threading
import torch


class MyModel(AfkImageModel):
    def __init__(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Convert the image to a tensor (values between 0 and 1)
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        # 为多线程并发环境下添加锁
        self.lock = threading.Lock()
        self.model = models.convnext_base(num_classes=2).eval()
        self.transform = transform

    def load_checkpoint(self, checkpoint):
        try:
            with self.lock:
                self.model.load_state_dict(torch.load(checkpoint))
                self.model.eval()
        except Exception as e:
            raise e

    def detect(self, img, label_only=False):

        with self.lock:
            # 根据输入类型构造输入张量
            if isinstance(img, list):
                input_image = torch.stack([self.transform(im) for im in img])
            else:
                input_image = self.transform(img).unsqueeze(0)

            # 进行一次推理
            with torch.no_grad():
                output: torch.Tensor = self.model(input_image).softmax(dim=1)

                if label_only:
                    output = output.argmax(dim=1)
                    return output.detach().numpy().astype(int)
                else:
                    return output.detach().numpy()
```

然后，我们的`model_impl.py`中修改对应的`get_server`方法：

```python
# model_impl.py
from afk_toolset_http import AfkServer
from my_conv_model import MyModel
def get_server():

    model = MyModel()
    model.load_checkpoint("/path/to/checkpoint")

    server = AfkServer("test_server")
    server.register_model(model)

    return server

```


### :train: 使用Docker运行服务（推荐）

为了更好的对接整个AntifakeAI套件生态，推荐使用Docker方式构建并运行模型。
TODO
```shell
docker build .
```

## :hammer: 构建仓库

本仓库基于`poetry`构建，如果想要手动构建`afk_toolset_http`包，执行以下命令：

```shell
poetry build .
```

