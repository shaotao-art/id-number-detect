from picamera import PiCamera
from time import sleep
import argparse

def get_photo():
    args = argparse.ArgumentParser()
    args.add_argument('--save_path', '-p', type=str, required=True, help="输入照片存储路径")
    args_dict = vars(args.parse_args())

    print('starting getting photo....')
    camera = PiCamera()
    camera.start_preview()
    sleep(5)
    camera.capture(args_dict['save_path'])
    camera.stop_preview()
    print(f'save photo to {args_dict["save_path"]}, done!')

if __name__ == "__main__":
    get_photo()
