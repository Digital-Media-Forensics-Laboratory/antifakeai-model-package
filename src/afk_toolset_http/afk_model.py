import torch
import torch.nn as nn
from typing import List, Union, Callable, Optional
from PIL import Image
import threading
import numpy.typing as npt
from io import BytesIO


class AfkModel:

    def convert_resource(self, content: bytes) -> Union[Image.Image]:
        raise NotImplementedError()

    def detect(
        self, img: Union[Image.Image, List[Image.Image], str, List[str]]
    ) -> npt.NDArray:
        raise NotImplementedError()

    def load_checkpoint(self, checkpoint: str) -> None:
        raise NotImplementedError()

    def uninstall(self) -> bool:
        raise NotImplementedError()

    def move_to_device(self, device: str):
        raise NotImplementedError()


class AfkImageModel(AfkModel):

    def __init__(
        self,
        model: nn.Module,
        transform: Callable[[Image.Image], torch.Tensor],
        device: str = "cuda",
    ):
        self.model: Optional[nn.Module] = model.to(device).eval()
        self.transform = transform
        self.device = device

        # 为多线程并发环境下添加锁
        self.lock = threading.Lock()

    def convert_resource(self, content) -> Image.Image:
        img = Image.open(BytesIO(content))
        return img

    def detect(
        self, img: Union[Image.Image, List[Image.Image]], label_only: bool = False
    ) -> npt.NDArray:
        """
        调用模型检测单张或多张图像
        :param img: 需要检测的图像
        :param label_only: 是否仅返回标签。``True`` 仅返回标签，``False`` 返回全部标签的置信度
        :return: 检测结果
        """

        # 检查模型是否存在
        if self.model is None:
            raise ValueError("model should not be None")

        with self.lock:
            # 根据输入类型构造输入张量
            if isinstance(img, Image.Image):
                input_image = self.transform(img).to(self.device).unsqueeze(0)
            elif isinstance(img, list):
                input_image = torch.stack([self.transform(im) for im in img]).to(
                    self.device
                )
            else:
                raise TypeError("input type not fitted")

            # 进行一次推理
            with torch.no_grad():
                output: torch.Tensor = self.model(input_image).softmax(dim=1)

                if label_only:
                    output = output.argmax(dim=1)
                    return output.detach().to("cpu").numpy().astype(int)
                else:
                    return output.detach().to("cpu").numpy()

    def load_checkpoint(self, checkpoint: str) -> None:
        """
        加载模型参数
        :param checkpoint: 模型参数路径
        :return:
        """
        try:
            with self.lock:
                self.model.load_state_dict(torch.load(checkpoint))
                self.model.eval()
        except Exception as e:
            raise e

    def move_to_device(self, device: str):
        with self.lock:
            self.device = device
            self.model = self.model.to(device)


class AfkTextModel(AfkModel):
    def __init__(
        self,
        model: nn.Module,
        transform: Callable[[Image.Image], torch.Tensor],
        device: str = "cuda",
    ):
        self.model: Optional[nn.Module] = model.to(device).eval()
        self.transform = transform
        self.device = device

        # 为多线程并发环境下添加锁
        self.lock = threading.Lock()

    def convert_resource(self, content: bytes) -> str:
        return content.decode("utf-8")

    def detect(self, text: str) -> npt.NDArray:
        raise NotImplementedError()

    def load_checkpoint(self, checkpoint: str) -> None:
        """
        加载模型参数
        :param checkpoint: 模型参数路径
        :return:
        """
        try:
            with self.lock:
                self.model.load_state_dict(torch.load(checkpoint))
                self.model.eval()
        except Exception as e:
            raise e

    def move_to_device(self, device: str):
        with self.lock:
            self.device = device
            self.model = self.model.to(device)


class AfkAudioModel(AfkModel):
    def __init__(
        self,
        model: nn.Module,
        transform: Callable[[Image.Image], torch.Tensor],
        device: str = "cuda",
    ):
        self.model: Optional[nn.Module] = model.to(device).eval()
        self.transform = transform
        self.device = device

        # 为多线程并发环境下添加锁
        self.lock = threading.Lock()

    def convert_resource(self, content: bytes) -> str:
        return content.decode("utf-8")

    def detect(self, text: str) -> npt.NDArray:
        raise NotImplementedError()

    def load_checkpoint(self, checkpoint: str) -> None:
        """
        加载模型参数
        :param checkpoint: 模型参数路径
        :return:
        """
        try:
            with self.lock:
                self.model.load_state_dict(torch.load(checkpoint))
                self.model.eval()
        except Exception as e:
            raise e

    def move_to_device(self, device: str):
        with self.lock:
            self.device = device
            self.model = self.model.to(device)
