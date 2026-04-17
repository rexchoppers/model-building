import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

transform = transforms.ToTensor()
train_data = datasets.MNIST('data', train=True, download=True, transform=transform)



# 64 images per batch for training stability
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
images, labels = next(iter(train_loader))
print(images.shape)
print(labels.shape)

# -- To show image
# image, label = train_data[0]
# plt.imshow(image.squeeze(), cmap='gray')
# plt.show()