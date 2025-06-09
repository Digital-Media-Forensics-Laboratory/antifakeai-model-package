from afk_toolset_http import AfkImageModel, AfkAudioModel
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
