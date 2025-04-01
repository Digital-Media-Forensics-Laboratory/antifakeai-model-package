# Model Toolkit of Anti-Fake AI

通过构建Docker镜像，将你的模型接入Anti-Fake AI

## 快速开始

1. 安装依赖
```shell
pip install afk_toolset_http
```

2. 移植模型
   
```python
# model.py

from afk_toolset_http import AfkImageModel

class My(AfkImageModel)
    pass

```
3. 注册模型

```python
# main.py
app = AfkServer()


app.run()

```

4. 构建镜像

```shell
docker build .
```

## 应用案例

为自己训练的convnext模型
