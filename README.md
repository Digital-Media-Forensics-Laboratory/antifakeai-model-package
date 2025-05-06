# Model Toolkit of Anti-Fake AI

**Integrate your own models into Anti-Fake AI by building a Docker image**

Languages:
- [English](README.md)
- [中文](README_zh.md)

Current base model support:

| Modality | Status     | Notes      |
|:--------:|:----------:|:----------:|
| Image    | :hammer:   | In progress |
| Audio    | :x:        | -          |
| Video    | :x:        | -          |
| Text     | :x:        | -          |

:hammer: : Under development  
:x: : Not supported

## :rocket: Quick Start

Template files are provided in the `example` directory. Based on the template, the following steps demonstrate how to add a custom `image detection` implementation.

1. Install dependencies

Use `pip` to install the `afk_toolset_http` package. The package file can be found in the releases of this project.
```shell
pip install afk_toolset_http.whl
```

2. Register your model

Implement the `get_server` function in `model_impl.py`. This function returns an `AfkServer` object, which is injected into the model service at runtime.

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
    model.load_checkpoint("/path/to/checkpoint")  # Load your custom model
    server = AfkServer("test_server")
    server.register_model(model)
    return server
```

3. Run the model

Once your model is ready, just launch the service to access it via HTTP requests. We have prepared `entry.py` for this.

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

Install and run the model service using `uvicorn`:
```shell
pip install "uvicorn[standard]"
uvicorn entry:app --host 0.0.0.0 --port 80
```
Now your model is accessible on port `80`!

### :pencil: Custom Model

We provide a basic image model wrapper class `AfkImageModel`, which suits common needs. If you require a highly customized model, define a `MyModel` class that implements the `detect` and `load_checkpoint` methods.

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
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        # Add a lock for thread safety
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
            # Construct input tensor based on input type
            if isinstance(img, list):
                input_image = torch.stack([self.transform(im) for im in img])
            else:
                input_image = self.transform(img).unsqueeze(0)

            # Perform inference
            with torch.no_grad():
                output: torch.Tensor = self.model(input_image).softmax(dim=1)

                if label_only:
                    output = output.argmax(dim=1)
                    return output.detach().numpy().astype(int)
                else:
                    return output.detach().numpy()
```

Then update the `get_server` function in `model_impl.py` accordingly:

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

### :train: Run the Service with Docker (Recommended)

To better integrate with the AntiFakeAI ecosystem, we recommend building and running your model using Docker.  
TODO
```shell
docker build .
```

## :hammer: Build the Package

This repo uses `poetry` for package management. To manually build the `afk_toolset_http` package, run:

```shell
poetry build .
```
