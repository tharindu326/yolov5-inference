from image_inference import ModelFileYOLOv5, InferenceYOLOV5
import cv2
import os
import yaml

def get_annotations(dataset_path_image, dataset_path_labels, gpu_device, models):
    for image in os.listdir(dataset_path_image):
        if image.endswith('.jpg') or image.endswith('.png') or image.endswith('.jpeg'):
            image_path = f'{dataset_path_image}{image}'
            txt_path = f'{dataset_path_labels}{image.split(".")[0]}.txt'
            inference = InferenceYOLOV5(gpu_device, models, overlay=False)
            img0 = cv2.imread(image_path)
            bboxes, scores, class_names, class_ids, image = inference.inferYOLOV5(img0)
            for i, name in enumerate(class_names):
                if name == "person":
                    h_img, w_img, channels = img0.shape
                    x = bboxes[i][0]
                    y = bboxes[i][1]
                    w = bboxes[i][2]
                    h = bboxes[i][3]
                    class_id = 10
                    with open(f'{txt_path}', 'a') as textfile:
                        textfile.write(
                            f'{class_id} {(x + w / 2) / w_img} {(y + h / 2) / h_img} {w / w_img} {h / h_img}')
                        textfile.write('\n')

def combine_classes(label_path, comine_classes_list):
    min_class_id = min(comine_classes_list)
    for txt in os.listdir(label_path):
        if txt.endswith('.txt'):
            print(f'{txt}=======================')
            txt_path = f'{label_path}{txt}'
            new_content = []
            with open(txt_path, 'r') as f:
                content = f.read().splitlines()
                for line in content:
                    class_id = line.split(' ')[0]
                    rest = line.split(' ')[1:]
                    if int(class_id) in comine_classes_list:
                        # new_line = rest.insert(0, min_class_id)
                        new_line = ' '.join([f'{min_class_id}'] + rest)
                    else:
                        if int(class_id) > min_class_id:
                            new_line = ' '.join([f'{(int(class_id) - 1)}'] + rest)
                        else:
                            new_line = line
                    new_content.append(new_line)
            os.remove(txt_path)
            with open(f'{txt_path}', 'w') as f:
                for item in new_content:
                    # write each item on a new line
                    f.write("%s\n" % item)



if __name__ == "__main__":
    model_directory = './model_data/'
    model_files = ModelFileYOLOv5(model_directory)
    target_gpu_device = '0'

    train = '/home/ubuntu/YOLOX/datasets/images/train2017/'
    val = '/home/ubuntu/YOLOX/datasets/images/val2017/'
    test = '/home/ubuntu/YOLOX/datasets/images/test2017/'
    
    train_labels = '/home/ubuntu/YOLOX/datasets/labels/train2017/'
    val_labels = '/home/ubuntu/YOLOX/datasets/labels/val2017/'
    test_labels = '/home/ubuntu/YOLOX/datasets/labels/test2017/'
    
    get_annotations(val, val_labels, target_gpu_device, model_files)
    get_annotations(test, test_labels, target_gpu_device, model_files)
    get_annotations(train, train_labels, target_gpu_device, model_files)

    # test = '/home/ubuntu/yolov5-inference/sample_dataset/'
    # test_labels = '/home/ubuntu/yolov5-inference/sample_dataset/'
    # get_annotations(test, test_labels, target_gpu_device, model_files)

    # yaml_data_file = 'coco128.yaml'
    # labels = yaml.safe_load(open(yaml_data_file, 'rb').read())['names']
    # combine_class_names = ['hemostat', 'scissor']
    # combine_classes_list = [labels.index(class_) for i, class_ in enumerate(combine_class_names)]
    # combine_classes(train_labels, combine_classes_list)



