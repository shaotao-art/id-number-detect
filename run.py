from pre_process import get_numbers
from get_column_num import get_each_number
from infer import recognize_numbers
import os
import argparse

def arg_parse():
    args = argparse.ArgumentParser()
    args.add_argument('--img_path', '-p', type=str, required=True, help="输入身份证照片路径")
    args_dict = vars(args.parse_args())
    return args_dict

def run(args_dict):
    tmp_dir = './tmp_run'
    id_numbers_path = os.path.join(tmp_dir, 'id_numbers.jpg')
    each_number_save_path = os.path.join(tmp_dir, 'numbers')

    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(each_number_save_path, exist_ok=True)

    get_numbers(file_path = args_dict['img_path'], save_path = id_numbers_path, use_rotate = False)
    get_each_number(path = id_numbers_path, save_path = each_number_save_path)
    res = recognize_numbers(each_number_save_path)
    return res

if __name__ == "__main__":
    args = arg_parse()
    run(args)