from .afk_data_model import DetectResult, DetectRequest, ServerMode, WeightMode
from .afk_model import AfkModel, AfkImageModel
from .afk_utility import AfkImageDownloader, AfkUtility
from typing import List
import numpy as np
from PIL import Image
import os


def create_random_image(width: int, height: int) -> Image.Image:
    # 生成一个随机的 (height, width, 3) 数组，表示 RGB 颜色值
    random_image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    # 将随机数组转为 PIL 图像
    random_image = Image.fromarray(random_image_array, "RGB")
    return random_image


class IAfkDetector:
    """AfkDetector 的接口类, 定义了检测器的基本行为"""

    def detect(self, request: List[DetectRequest]) -> List[DetectResult]:
        raise NotImplementedError()

    def detect_single(self, request: DetectRequest) -> DetectResult:
        """
        基于单请求的检测
        Args:
            request (DetectRequest): 需要进行检测的请求
        Returns:
            DetectResult: 检测结果
        Raises:
            NotImplementedError: 未实现的方法
        """

        raise NotImplementedError()

    def append_model(self, model: AfkModel):
        """
        Detects a single instance based on the provided request.
        Args:
            model (AfkModel): The model to be appended to the dector.
        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        raise NotImplementedError()


class AfkDetector(IAfkDetector):
    """
    AfkDetector is a class responsible for managing and utilizing multiple AfkModels to detect AIGC image in given requests.

    Attributes:
        models (List[AfkModel]): A list to store the models used for detection.
        server_mode (ServerMode): The mode in which the server operates, e.g., "align".
        weight_mode (WeightMode): The mode to determine how to weight the results from different models, e.g., "average".

    Methods:
        append_model(model: AfkModel):
            Appends a model to the list of models, ensuring it matches the type of existing models if in align mode.

        detect(request: List[DetectRequest]) -> List[DetectResult]:
            Detects anomalies in the given requests using the stored models and returns the results.
    """

    def __init__(self, server_mode: ServerMode, weight_mode: WeightMode):
        self.models: List[AfkModel] = []
        self.server_mode = server_mode
        self.weight_mode = weight_mode

    @staticmethod
    def __is_same_type(lhs, rhs) -> bool:
        """
        Check if both lhs and rhs are instances of AfkImageModel.

        Args:
            lhs: The left-hand side object to compare.
            rhs: The right-hand side object to compare.

        Returns:
            bool: True if both lhs and rhs are instances of AfkImageModel, False otherwise.
        """
        if isinstance(lhs, AfkImageModel) and isinstance(rhs, AfkImageModel):
            return True
        else:
            return False

    def append_model(
        self,
        model: AfkModel,
    ):

        if (
            self.server_mode == "align"
            and self.models
            and not self.__is_same_type(self.models[0], model)
        ):
            raise ValueError("Server is in align mode but heterogeneous model is find.")
        self.models.append(model)

    def detect_single(self, request: DetectRequest) -> DetectResult:
        """
        对单个请求进行检测,以单个的形式送入检测器
        :param request: 需要进行检测的请求
        :return: 请求的检测结果
        """
        if self.server_mode == "align":

            if os.path.isfile(request.resource_url):
                with open(request.resource_url, "rb") as f:
                    binary_content = f.read()
            else:

                binary_content = AfkUtility.download_resource(request.resource_url)

            detect_content = self.models[0].convert_resource(binary_content)

            raw_results = [m.detect(detect_content)[0] for m in self.models]

            if self.weight_mode == "average":
                detect_result = np.mean(raw_results, axis=0)[0]

            else:
                raise NotImplementedError(f"{self.weight_mode} is not implemented")

        else:
            raise NotImplementedError(f"{self.server_mode} is not implemented")

        return DetectResult(taskId=request.task_id, result=detect_result)

    def detect(self, request: List[DetectRequest]) -> List[DetectResult]:
        """
        对传递过来的请求进行检测,以批量的形式送入检测器
        :param request: 需要进行检测的所有请求
        :return: 所有请求的检测结果,顺序与请求保持一致
        """

        if self.server_mode == "align":

            # 通过http下载内容
            contents = [AfkUtility.download_resource(r.resource_url) for r in request]

            # 将下载的二进制内容转换为符合模型要求的格式,如 PIL.Image.Image
            model = self.models[0]  # 对齐模式下可以任意选择一个模型作类型转换器
            detect_contents = [model.convert_resource(content) for content in contents]

            # 获取每个模型的检测结果 -> List[(b, dim1, dim2)]
            raw_results = [m.detect(detect_contents) for m in self.models]

            # 根据权重模式来对所有模型处理的结果汇总,形成最终结果
            if self.weight_mode == "average":
                # 对每个维度取平均
                detect_results = np.mean(raw_results, axis=0)
            else:
                # TODO: 目前只实现了平均模式
                raise NotImplementedError(f"{self.weight_mode} is not implemented")

            # 返回检测结果,顺序与请求一致
            results = [
                DetectResult(taskId=r.task_id, result=detect_results[i][0])
                for i, r in enumerate(request)
            ]
            return results
