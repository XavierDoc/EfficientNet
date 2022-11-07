import torch
import torchvision
from cifar10 import build_cifar10
from EfficientNet import efficientnet_b0
import matplotlib.pyplot as plt
import numpy as np

build_data = build_cifar10(False)
train_dataset, val_dataset = build_data

train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=32,
                                            pin_memory=True)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=32,
                                            pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(val_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = efficientnet_b0().to(device)
model.load_state_dict(torch.load(''))#need pt file path

outputs = model(images)  

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))