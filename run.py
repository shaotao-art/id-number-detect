
from get_photo import get_photo
from pre_process import get_numbers
from getColumnNum import get_each_number
from infer import recognize_numbers

def run():
    origin_file_path = './TEST_IMG/test1.jpg'
    id_numbers_path = './TEST_IMG/out1.jpg'
    each_number_save_path = './number_cut_out'
    get_photo(file_name = origin_file_path)
    get_numbers(file_path = origin_file_path, save_path = id_numbers_path, use_rotate = False)
    numbers_lst = get_each_number(path = id_numbers_path, save_path = each_number_save_path)
    recognize_numbers(each_number_save_path)

if __name__ == "__main__":
    run()