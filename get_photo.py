from picamera import PiCamera
from time import sleep
# test pull requests

def get_photo(file_name = 'one.jpg'):
    print('starting getting photo....')
    camera = PiCamera()

    camera.start_preview()
    sleep(5)
    camera.capture(file_name)
    camera.stop_preview()
    print(f'save photo to {file_name}, done!')
