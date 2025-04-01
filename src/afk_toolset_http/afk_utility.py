import requests
from PIL import Image
from io import BytesIO


class AfkUtility:
    @staticmethod
    def download_resource(resource_url: str) -> bytes:
        response = requests.get(resource_url)
        return response.content


class AfkImageDownloader:
    @staticmethod
    def download(url: str) -> Image.Image:
        img_content = AfkUtility.download_resource(url)
        img = Image.open(BytesIO(img_content))
        return img
