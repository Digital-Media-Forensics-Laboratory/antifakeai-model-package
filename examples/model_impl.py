import torchvision
from torchvision import transforms
from afk_toolset_http import AfkServer, AfkImageModel


def get_server():
    conv_model = torchvision.models.convnext_base(num_classes=2).to("cuda").eval()
    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),  # Resize image to 256x256
            transforms.ToTensor(),  # Convert the image to a tensor (values between 0 and 1)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    model = AfkImageModel(conv_model, transform)
    model.load_checkpoint("/path/to/checkpoint")
    server = AfkServer("test_server")
    server.register_model(model)

    return server
