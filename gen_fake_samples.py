import numpy as np
from PIL import Image, ImageFont, ImageDraw
import os
from matplotlib import pyplot as plt

def gen_background(size, mu, sigma):
    '''
    generate a background grayscale image
    size:  image size
    mu:    bright mean
    sigma: bright variance

    return Image
    '''

    pixel_array = sigma * np.random.randn(size[1], size[0]) + mu
    pixel_array[pixel_array > 255] = 255
    pixel_array[pixel_array < 0  ] = 0
    pixel_array = pixel_array.astype(np.uint8)

    return Image.fromarray(pixel_array, 'L')


def rend_char_img(char, angle, img_size, font_size, font_type):
    '''
    rend a character image
    char:      character to be rend
    angle:     character rotate angle
    img_size:  size of output image
    font_size: font size
    font_type: font type

    return character image, white background and black forecolor
    '''

    img = Image.new('L', img_size, 0)
    font = ImageFont.truetype(font_type, size = font_size)
    char_size = font.getsize(char)

    drawer = ImageDraw.Draw(img)
    drawer.text(((img_size[0] - char_size[0]) / 2, (img_size[1] - char_size[1]) / 2), char, font = font, fill = 255)
    del drawer
    
    img = img.rotate(angle)

    return img


def clip_box(box, edge, img_size):
    '''
    make a square clip box base on real box + edge
    box: real box
    img_size: clip box must less than img_size
    '''

    left, top, right, bottom = box
    left   -= edge[0]
    top    -= edge[1]
    right  += edge[2]
    bottom += edge[3]
    w = right - left
    h = bottom - top
    if (w > h):
        top    -= (w - h) / 2
        bottom += (w - h) / 2
    elif (w < h):
        left  -= (h - w) / 2
        right += (h - w) / 2
    if left < 0:
        left = 0
    if top < 0:
        top = 0
    if right > img_size[0]:
        right = img_size[0]
    if bottom > img_size[1]:
        bottom = img_size[1]

    out_box = (left, top, right, bottom)

    #print("box = ",  str(box))
    #print("edge = ", str(edge))
    #print("img_size = ", str(img_size))

    return out_box


def randomize_and_crop_char_img(char_img, mu, sigma, edge):
    '''
    add random to character image
    mu:    mena of character stroke is 255 * (1.0 - mu)
    sigma: variance of character stroke is 255 * sigma
    edge:  a tuple, 4 side edge after crop
    '''

    randmizer = sigma * np.random.randn(char_img.size[1], char_img.size[0]) + mu
    pix = np.array(char_img)
    
    #disable halftone, otherwise a white edge may sourround the character
    pix[pix > 0] = 255 
    
    pix = pix * randmizer
    pix[pix > 255] = 255
    pix[pix < 0  ] = 0

    pix = pix.astype(int)   
    pix = 255 - pix    #swith forecolor and background color
    pix = pix.astype(np.uint8)

    box = clip_box(char_img.getbbox(), edge, char_img.size)
    croped = Image.fromarray(pix, 'L').crop(box)
    mask = char_img.crop(box)
    
    return croped, mask


def merge_char_and_background(back_img, char_img, mask):
    '''
    merge char image onto back image

    back_img must char_img mask must be the same size
    '''
    
    back_arr = np.array(back_img)
    char_arr = np.array(char_img)
    mask_arr = np.array(mask)

    #print(back_arr.shape, char_arr.shape, mask_arr.shape)

    bit_mask = (mask_arr > 0).astype(int)
    pix = char_arr * bit_mask + back_arr * (1 - bit_mask)
    pix[pix > 255] = 255
    pix[pix < 0  ] = 0
    pix = pix.astype(np.uint8)

    merged_img = Image.fromarray(pix, 'L')
    
    return merged_img


def gen_char_img(char, angle, back_mean, back_var, fore_mean, fore_var, font_path, out_size = 32, margin = (0, 0, 0, 0)):
    '''
    char:    the character to be rended
    angle:   angle of the character rotation
    
    back_mean: white background bright mean, back_mean - back_var should be larger than fore_mean + fore_var
    back_var:  white bakcground bright variance
    
    fore_mean: black forecolor bright mean, back_mean - back_var should be larger than fore_mean + fore_var
    fore_var:  black forecolor bright variance 

    out_size: output image size, a integer, the real output image size is (out_size, out_size)

    margin:  4 side (left, top, right ,bottom) spaces
    '''
  
    origi_char_img = rend_char_img(char, angle, (out_size*4, out_size*4), out_size*3, font_path)
       
    char_img_with_rand, mask = randomize_and_crop_char_img(origi_char_img, (255.0 - fore_mean) / 255.0,  fore_var / 255.0, margin)
    
    background_img = gen_background(char_img_with_rand.size, back_mean, back_var)

    merged_img = merge_char_and_background(background_img, char_img_with_rand, mask)

    out_img = merged_img.resize((out_size, out_size), Image.BILINEAR)
    
    return out_img

# 0 纯黑 1 白色
def gen_samples(char, num_samples, output_dir, angle_high=7, margin_lst=[6, 7, 8, 9]):
    for _ in range(num_samples):
        angle = np.random.randint(low=0, high=angle_high)
        margin = int(np.random.choice(margin_lst, size=(1)))
        img = gen_char_img(char=char, 
                    angle=angle, 
                    back_mean=0, 
                    back_var=0, 
                    fore_mean=255, 
                    fore_var=0, 
                    font_path='./fonts/ocr-font.ttf', 
                    out_size = 32, 
                    margin = (margin, margin, margin, margin))
        plt.imsave(os.path.join(output_dir ,f'{np.random.randint(10000)}.png'), img)
    print(f'generating char: {char}, num samples gen: {num_samples}')


def gen_dataset(char_lst = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'X'], num_samples_for_each = 2000):
    for x in char_lst:
        fold_path = f'./data_gen/char_{x}'
        os.makedirs(fold_path, exist_ok=True)
        gen_samples(x, num_samples_for_each, fold_path)


if __name__ == "__main__":
    gen_dataset()