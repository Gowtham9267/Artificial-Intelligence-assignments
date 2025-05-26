pip install torch torchvision pillow
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import copy

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image loader and preprocessor
def load_image(img_path, max_size=400):
    image = Image.open(img_path).convert("RGB")
    size = max(max(image.size), max_size)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3, :, :]),  # Ensure RGB
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

# Image unloader for display or save
def im_convert(tensor):
    image = tensor.cpu().clone().detach().squeeze(0)
    image = transforms.Normalize(
        [-2.118, -2.036, -1.804],
        [4.367, 4.464, 4.444]
    )(image)
    image = torch.clamp(image, 0, 1)
    return transforms.ToPILImage()(image)

# Content and Style loss layers
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def gram_matrix(self, input):
        b, c, h, w = input.size()
        features = input.view(c, h * w)
        G = torch.mm(features, features.t())
        return G.div(c * h * w)

    def forward(self, x):
        G = self.gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

# Load VGG19 model
def get_model_and_losses(content_img, style_img):
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    content_losses = []
    style_losses = []

    model = nn.Sequential().to(device)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_losses.append(ContentLoss(target))

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_losses.append(StyleLoss(target_feature))

    for loss in content_losses + style_losses:
        model.add_module("loss", loss)

    return model, style_losses, content_losses

# Run style transfer
def run_style_transfer(content_img, style_img, input_img, num_steps=300, style_weight=1e6, content_weight=1):
    model, style_losses, content_losses = get_model_and_losses(content_img, style_img)
    optimizer = torch.optim.LBFGS([input_img.requires_grad_()])

    print("Optimizing...")
    run = [0]
    while run[0] <= num_steps:
        def closure():
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(l.loss for l in style_losses)
            content_score = sum(l.loss for l in content_losses)
            loss = style_score * style_weight + content_score * content_weight
            loss.backward()
            run[0] += 1
            return loss

        optimizer.step(closure)

    return input_img

# Example usage
content_img = load_image("your_photo.jpg")
style_img = load_image("your_style.jpg")
input_img = content_img.clone()

output = run_style_transfer(content_img, style_img, input_img)
output_image = im_convert(output)
output_image.save("styled_output.jpg")
