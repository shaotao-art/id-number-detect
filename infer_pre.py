import torch
from torchvision import transforms
import pre_process
from src.model import NNet


T = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

model_save_path = './ckp/model.ckp'
checkpoint = torch.load(model_save_path, map_location='cpu')
model = NNet()
model.load_state_dict(checkpoint['model'])
x = pre_process.preprocess('./sample1.jpg')
x = x.unsqueeze(0).unsqueeze(0)
pred = model(x).argmax()
print(f'prediiction: {pred}')