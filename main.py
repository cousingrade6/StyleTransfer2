import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from utils import *
from models import *

from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

style_path = "E:/ArtisticStyle/style/Vangogh.jpg"
style_image = read_image(style_path).to(device)
plt.ion()
plt.figure()
imshow(style_image, title='Style Image')


# 损失函数
def content_loss(y_hat, y):
    return torch.square(y_hat - y.detach()).mean()


def gram_matrix(y):
    b, num_channels, h, w = y.size()
    features = y.view(b, num_channels, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (num_channels * h * w)


def style_loss(y_hat, gram_y):
    return torch.square(gram_matrix(y_hat) - gram_y.detach()).mean()


def tv_loss(y_hat):
    return (torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :])) +
            torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1])))


# rebuild vgg
# import VGG net
vgg_net = models.vgg16(pretrained=True).features.eval()
print(vgg_net)


def get_layers_name_list(net):
    i, j = 0, 1  # increment every time we see a conv
    name_list = []
    for layer in net.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv{}_{}'.format(j, i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu{}_{}'.format(j, i)
        elif isinstance(layer, nn.MaxPool2d):
            j += 1
            name = 'pool{}_{}'.format(j, i)
            i = 0
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn{}_{}'.format(j, i)
        else:
            continue
        name_list.append(name)
    return name_list


name_list = get_layers_name_list(vgg_net)
print(name_list)

content_layers_names = ['conv3_3']
style_layers_names = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']
content_layers_list = []
style_layers_list = []

for layer in content_layers_names:
    content_layers_list.append(name_list.index(layer))
for layer in style_layers_names:
    style_layers_list.append(name_list.index(layer))
print(style_layers_list)

net = nn.Sequential(*[vgg_net[i] for i in
                      range(max(content_layers_list + style_layers_list) + 1)])


def extract_feature(x, layer_list):
    feature = []
    for i in range(len(net)):
        x = net[i](x)
        if i in layer_list:
            feature.append(x)
    return feature


# 载入数据集
batch_size = 8
width = 256

data_transform = transforms.Compose([
    transforms.Resize(width),
    transforms.CenterCrop(width),
    transforms.ToTensor(),
    tensor_normalizer,
])

dataset = LoadData('E:/ArtisticStyle/archive/coco2017', transform=data_transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device='cuda'))

# transfors_net 为待训练的网络
transforms_net = TransformNet(32)

style_weight = 1e5
content_weight = 1
tv_weight = 1e-6

optimizer = torch.optim.Adam(transforms_net.parameters(), 1e-3)
transforms_net.train()

n_batch = len(data_loader)


def train():
    smooth_content_loss = Smooth()
    smooth_style_loss = Smooth()
    smooth_tv_loss = Smooth()
    smooth_loss = Smooth()
    style_features = extract_feature(style_image, style_layers_list)
    style_y_gram = [gram_matrix(y) for y in style_features]
    with tqdm(enumerate(data_loader), total=n_batch) as pbar:
        for batch, content_images in pbar:
            optimizer.zero_grad()
            content_images = content_images.to(device)
            transformed_images = transforms_net(content_images)
            transformed_images = transformed_images.clamp(-3, 3)

            # content loss
            content_features = extract_feature(content_images, content_layers_list)
            transformed_features = extract_feature(transformed_images, content_layers_list)
            content_l = sum([content_loss(y_hat, y) * content_weight for y_hat, y in zip(
                transformed_features, content_features
            )])

            # style loss
            transformed_features = extract_feature(transformed_images, style_layers_list)
            style_l = sum([style_loss(y_hat, gram_y) * style_weight for y_hat, gram_y in zip(
                transformed_features, style_y_gram
            )])

            # tv loss
            tv_l = tv_weight * tv_loss(transformed_images)

            loss = style_l + content_l + tv_l
            loss.backward()
            optimizer.step()

            smooth_content_loss += content_l.item()
            smooth_style_loss += style_l.item()
            smooth_tv_loss += tv_l.item()
            smooth_loss += loss.item()

            s = f'Content: {smooth_content_loss:.2f} '
            s += f'Style: {smooth_style_loss:.2f} '
            s += f'TV: {smooth_tv_loss:.4f} '
            s += f'Loss: {smooth_loss:.2f}'
            if batch % 500 == 0:
                s = '\n' + s
                save_debug_image(style_image, content_images, transformed_images,
                                 f"debug/s2_{batch}.jpg")

            pbar.set_description(s)
    torch.save(transforms_net.state_dict(), 'transform_net.pth')

transform_net = TransformNet(32)
transform_net.load_state_dict(torch.load('transform_net.pth'))
def test():
    content_img = read_image('E:/ArtisticStyle/content.jpg').to(device)
    output_img = transform_net(content_img)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    imshow(style_image, title='Style Image')

    plt.subplot(1, 3, 2)
    imshow(content_img, title='Content Image')

    plt.subplot(1, 3, 3)
    imshow(output_img.detach(), title='Output Image')

    plt.ioff()
    plt.show()

# train()
test()
