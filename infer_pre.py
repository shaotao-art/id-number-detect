import torch
from torchvision import transforms
from proj_template.src.model import NNet
from matplotlib import pyplot as plt
from PIL import Image


def get_T():
    T = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    return T


def get_model(model_save_path = './ckp/model.ckp'):
    checkpoint = torch.load(model_save_path, map_location='cpu')
    model = NNet()
    model.load_state_dict(checkpoint['model'])
    return model


def prepare_imgs(T):
    out_lst = []
    root_path = './number_cut_out'
    import os
    file_num = len(os.listdir(root_path))
    assert file_num == 18, 'not enough img to infer'
    for i in range(18):
        file_path = os.path.join(root_path, f"img_{i}.jpg")
        img = Image.open(file_path)
        img = T(img)
        out_lst.append(img.unsqueeze(0))
    return torch.cat(out_lst)


def infer(model, img):
    pred = model(img)
    res = pred.argmax(1)
    return res


if __name__ == "__main__":
    model = get_model()
    T = get_T()
    imgs = prepare_imgs(T)
    res = infer(model, imgs)
    print(res)