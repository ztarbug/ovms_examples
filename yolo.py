import ovmsclient
import pprint as pp
import cv2
import numpy as np
import json
import sys
import os

output_path = "output"
image_path = "sample_images"

def main():
    check_project_structure()

    for path in os.listdir(image_path):
        if os.path.isfile(image_path + "/" + path):
            print(image_path + "/" + path)
            do_inferencing(image_path + "/" + path)

def check_project_structure():
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    else:
        if not os.path.isdir(output_path):
            print("output folder is file, can't output results. Exit")
            sys.exit()

    if not os.path.exists(image_path):
        os.mkdir(image_path)
    else:
        if not os.path.isdir(image_path):
            print("image folder is file, can't read images. Exit")
            sys.exit()

    if not os.path.exists("coco_classes.json"):
        print("coco classes don't exist. Exit.")
        sys.exit()
 

def do_inferencing(filepath):
    client = ovmsclient.make_grpc_client("localhost:9001")
    mdata = client.get_model_metadata(model_name = "yolov7")

    classes = {value['id'] - 1: value['name'] for value in json.load(open('coco_classes.json', 'r')).values()}
    # If model has only one input, get its name
    input_name = next(iter(mdata["inputs"]))

    img_raw = cv2.imread(filepath, cv2.IMREAD_COLOR)

    img_h, img_w = img_raw.shape[0:2]
    img = cv2.resize(img_raw.astype(np.float32), (640, 640))
    print(img.shape)
    column_order = [0,1,2]
    img = img.transpose(2,0,1)
    print(img.shape)
    img_r_h, img_r_w = img.shape[0:2]

    scale_h = img_h/img_r_h
    scale_w = img_w/img_r_w
    print(str(scale_h) + ", " + str(scale_w))

    input_data = np.expand_dims(img, 0).astype(np.float32)
    inputs = {input_name: input_data} 

    results = client.predict(inputs=inputs, model_name="yolov7")

    pp.pprint(results)

    #TODO

main()