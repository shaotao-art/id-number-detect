import torch
from torchvision import transforms
from number_recog.src.model import NNet
from PIL import Image
import os


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
    model.eval()
    return model


def prepare_imgs(T, root_path):
    out_lst = []
    file_num = len(os.listdir(root_path))
    assert file_num == 18, 'not enough img to infer'
    for i in range(18):
        file_path = os.path.join(root_path, f"img_{i}.jpg")
        img = Image.open(file_path)
        img = T(img)
        out_lst.append(img.unsqueeze(0))
    return torch.cat(out_lst)


def infer(model, img):
    with torch.no_grad():
        pred = model(img)
        res = pred.argmax(1)
    return res

def recognize_numbers(root_path):
    model = get_model()
    T = get_T()
    imgs = prepare_imgs(T, root_path)
    res = infer(model, imgs)
    assert res.shape[-1] == 18, f"should get 18 numbers, but get {res.shape}"
    lst = [res[i].item() for i in range(18)]
    print('----- final res -----')
    print(f'your id number is {lst}')
    print('----- final res -----')
    return lst

if __name__ == "__main__":
    recognize_numbers('./number_cut_out')

    ##### test for single image ######
    # img = Image.open('/Users/starfish/Desktop/python_proj/ML-funny-projects/rasp/tmp_run/numbers/img_1.jpg')
    # T = get_T()
    # img = T(img)
    # model = get_model()
    # res = infer(model, img)
    # print(res)
