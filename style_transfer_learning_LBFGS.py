import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import os

# change the path model downloaded.
os.environ['TORCH_HOME'] = './models'

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import copy

device_num = 2
device = torch.device(device_num if torch.cuda.is_available() else "cpu")

imsize = 512

loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()])


def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)  # add fake batch. expand dims.
    return image.to(device, torch.float)


style_img = image_loader("/home/user01/data_ssd/LeeJongHyeok/pytorch_project/style_transfer/images/picasso.jpg")
content_img = image_loader("/home/user01/data_ssd/LeeJongHyeok/pytorch_project/style_transfer/images/dancing.jpg")


def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze()
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.show()


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()

        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    N, C, H, W = input.size()
    features = input.view(N * C, H * W)  # 어차피 n 은 1이여야 함, style 이미지는 한장이기 때문에.
    G = torch.mm(features, features.t())  # (Nl * Ml)(Nl * Ml) = (Nl * Nl)

    return G.div(N * C * H * W)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()

        self.target = gram_matrix(target_feature).detach()  # grad detach

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        # self.loss = torch.reciprocal(4 * torch.pow(features.size()[0], 2) * torch.pow(features.size()[1], 2)) * self.loss

        return input


cnn = models.vgg19(pretrained=True).features.to(device).eval()  # evaluation mode.

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()

        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img,
                               content_layers=content_layers_default, style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)

        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)  # /?????

        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)

        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)

        else:
            raise RuntimeError("Unrecognized layer")

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()  # at 'name' layer output.
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    # NOTICE : if at i step from the end meet the content loss or style loss then break. and 'i' mean the last step index.
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]  # trim the model.

    return model, style_losses, content_losses


# input_img = content_img.clone()

input_img = torch.randn(content_img.size()).to(device)

imshow(input_img, title="Input image")


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    print([input_img.requires_grad_()])
    # NOTICE:  input 이미지에 대해서만 gradient 를 계산. loss는 기존 gradient load 에서 detach 되어있으므로 loss.bacward() 후 이 optimizer 를 그 loss 로 update 하면 이미지의 데이터가 업데이트됨.
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_image, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                                                     style_image, content_img)

    optimizer = get_input_optimizer(input_img)

    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)  # make data to 0 to 1

            optimizer.zero_grad()
            model(input_img)  # NOTICE: loss's forward also be run.

            style_score = 0
            content_score = 0

            for sl in style_losses:  # conv_1, 'conv_2', 'conv_3', 'conv_4', 'conv_5
                style_score += sl.loss  # loss already calculated at  //model(input_img)

            for cl in content_losses:  # conv4
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))

            return loss

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

    return input_img


output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)


imshow(content_img, title='content Image')
imshow(style_img, title='style Image')
imshow(output, title='Output Image')

