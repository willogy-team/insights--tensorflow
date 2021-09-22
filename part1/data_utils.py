import os
from shutil import copy2

import xml.etree.ElementTree as ET
import scipy.io


def read_xml_file(xml_file: str):
    '''
    Read the Pascal VOC XML file

    Arguments:
      - xml_file(str): Absolute path to an XML file of the Stanford Dogs dataset

    Returns:
      - filename(str): The file name of the corresponding sample image
      - list_of_all_boxes(str): List of all bounding boxes extracted from the XML file (there is only one bounding box in each file of this dataset)
    '''
    tree = ET.parse(xml_file)
    print("[1] tree: ", tree, type(tree))
    root = tree.getroot()
    print("[2] root:", root, type(root))

    list_of_all_boxes = []

    filename = root.find("filename").text
    print("[3] root.iter('object'): ", root.iter("object"), type(root.iter("object")))
    for boxes in root.iter("object"): # Iter over the ```object``` tag
        y_min, x_min, y_max, x_max = None, None, None, None

        y_min = int(boxes.find("bndbox/ymin").text) # slask ('/') is to access at lower levels
        x_min = int(boxes.find("bndbox/xmin").text) # bndbox is the parent tag of ymin, ...
        y_max = int(boxes.find("bndbox/ymax").text)
        x_max = int(boxes.find("bndbox/xmax").text)

        list_of_single_boxes = [x_min, y_min, x_max, y_max]
        list_of_all_boxes.append(list_of_single_boxes)
    # Here, because there is only one bbox per annotation file
    # => The for loop only has one iteration

    return filename, list_of_all_boxes

def read_mat_file(mat_file: str, num_classes, num_files):
    '''
    Read the .mat file

    Arguments:
      - mat_file(str): Absolute path to a .mat file
      - num_classes(int): The number of classes/categories to extract. There are 120 classes/categories in the Stanford Dogs dataset, but we can only choose some first classes/categories to train for a quick introduction.
      - num_files(int): The number of files in each class to extract. We only get a small number of files for a quick introduction.
    
    Returns:
      - dic_of_used_files(dictionary) - a dictionary with:
        - key: a string represents each class/category
        - value: a list of sample file names that belong to the class specified in the key
    '''
    mat_info = scipy.io.loadmat(mat_file)

    file_list = mat_info["file_list"]
    print('[1] file_list: ', file_list, type(file_list))

    print('[2] file_list[0, 0][0]: ', str(file_list[0, 0][0]), type(str(file_list[0, 0][0])))

    dic_of_used_files = {}
    cnt = 0
    for id, file in enumerate(file_list):
        # if id == num_files:
        #     break
        # print(str(file[0][0]))
        cur_class, file_path = file[0][0].split("/")
        if cur_class not in dic_of_used_files:
            cnt += 1
            if cnt > num_classes:
                break
            dic_of_used_files[cur_class] = []
        if len(dic_of_used_files[cur_class]) < num_files:
            dic_of_used_files[cur_class].append(file_path)

    return dic_of_used_files

def split_data_into_dirs(input_data_path, output_data_path, num_classes, train_samples, test_samples):
    '''
    Split data into train directory and test directory

    Arguments:
      - input_data_path(str): Path to the Stanford Dogs dataset forlder
      - output_data_path(str): Path to the output dataset after splitting. In this folder will be two more subfolders: "train" and "test". These two folders are automatically created.
      - num_classes(int): The number of classes/categories to extract
      - train_samples(int): The number of files in each class to extract for the "train" folder
      - test_samples(int): The number of files in each class to extract for the "test" folder
    '''
    input_images_path = os.path.join(input_data_path, "Images")
    input_annotations_path = os.path.join(input_data_path, "Annotation")
    input_lists_path = os.path.join(input_data_path, "lists")
    train_mat_file_path = os.path.join(input_lists_path, "train_list.mat")
    test_mat_file_path = os.path.join(input_lists_path, "test_list.mat")

    train_dic_of_used_files = read_mat_file(train_mat_file_path, num_classes, train_samples)
    test_dic_of_used_files = read_mat_file(test_mat_file_path, num_classes, test_samples)
    class_names = train_dic_of_used_files.keys()

    # Create dirs
    os.makedirs(output_data_path, exist_ok=True)
    os.makedirs(os.path.join(output_data_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_data_path, 'test'), exist_ok=True)

    for class_name in class_names:
        os.makedirs(os.path.join(output_data_path, 'train', class_name), exist_ok=True)
        os.makedirs(os.path.join(output_data_path, 'test', class_name), exist_ok=True)

    for class_name, list_of_files in train_dic_of_used_files.items():
        for file_name in list_of_files:
            in_path = os.path.join(input_images_path, class_name, file_name)
            out_path = os.path.join(output_data_path, "train", class_name, file_name)
            copy2(in_path, out_path)

    for class_name, list_of_files in test_dic_of_used_files.items():
        for file_name in list_of_files:
            in_path = os.path.join(input_images_path, class_name, file_name)
            out_path = os.path.join(output_data_path, "test", class_name, file_name)
            copy2(in_path, out_path)


if __name__ == "__main__":
    xml_file_path = "/media/data-huy/dataset/StanfordDogs/Annotation/n02085620-Chihuahua/n02085620_7"
    name, boxes = read_xml_file(xml_file_path)
    print("[*] filename: ", name)
    print("[*] boxes: ", boxes)

    mat_file_path = "/media/data-huy/dataset/StanfordDogs/lists/file_list.mat"
    dic_of_used_files = read_mat_file(mat_file_path, 10, 1000)
    # print('[*] dic_of_used_files: ', dic_of_used_files)

    print("[*] keys: ", len(dic_of_used_files.keys()))
    for key, value in dic_of_used_files.items():
        print("[*] key: ", key)
        print(value)

    input_data_path = "/media/data-huy/dataset/StanfordDogs"
    output_data_path = "/media/data-huy/dataset/StanfordDogs/train_val_test"
    split_data_into_dirs(input_data_path, output_data_path, 3, 100, 30)
