import argparse, random
from pathlib import Path

import cv2, os, pathlib,sys
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.slicing import get_auto_slice_params
from sahi.utils.yolov8 import download_yolov8s_model

project_folder_path = pathlib.Path(__file__).resolve().parents[0]
# sys.path.append(str(project_folder_path))

default_weights_path = os.path.join(project_folder_path, 'models', 'yolov8x-oiv7.pt')



def yolo_detect_windows_single(input_image, slice_h=512, slice_w=512):
    weights_path = default_weights_path
    
    
     # Check weights path
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Source path '{weights_path}' does not exist.")


    detection_model = AutoDetectionModel.from_pretrained(model_type='yolov8',
                                                         model_path=weights_path,
                                                         confidence_threshold=0.1,
                                                         device='gpu')
    

    [height, width, _] = input_image.shape


    # Prepare a square image for inference
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = input_image

    
    results = get_sliced_prediction(image,
                                    detection_model,
                                    slice_height=slice_h,
                                    slice_width=slice_w,
                                    overlap_height_ratio=0.2,
                                    overlap_width_ratio=0.5)
    object_prediction_list = results.object_prediction_list

    windows = []
    trees = []
 
    for ind, _ in enumerate(object_prediction_list):
        box = object_prediction_list[ind].bbox.minx, object_prediction_list[ind].bbox.miny, \
            object_prediction_list[ind].bbox.maxx, object_prediction_list[ind].bbox.maxy
        clss = object_prediction_list[ind].category.name

        score = object_prediction_list[ind].score.value
        if clss == "Window":
            windows.append((box,score))
        elif clss == "Tree":
            trees.append((box,score))

       
    return windows, trees


def main(input_image_path, weights_path, view_img=False, save_img=False, exist_ok=False):
    """
    Run object detection on a video using YOLOv8 and SAHI.

    Args:
        weights (str): Model weights path.
        source (str): Video file path.
        view_img (bool): Show results.
        save_img (bool): Save results.
        exist_ok (bool): Overwrite existing files.
    """

    # Check image path
    if not Path(input_image_path).exists():
        raise FileNotFoundError(f"Source path '{input_image_path}' does not exist.")
    
     # Check weights path
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Source path '{weights_path}' does not exist.")


    detection_model = AutoDetectionModel.from_pretrained(model_type='yolov8',
                                                         model_path=weights_path,
                                                         confidence_threshold=0.1,
                                                         device='gpu')
    
     # Read the input image
    original_image: np.ndarray = cv2.imread(input_image_path)
    [height, width, _] = original_image.shape

    # Prepare a square image for inference
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image

    
    results = get_sliced_prediction(image,
                                    detection_model,
                                    slice_height=512,
                                    slice_width=512,
                                    overlap_height_ratio=0.2,
                                    overlap_width_ratio=0.2)
    object_prediction_list = results.object_prediction_list

    boxes_list = []
    clss_list = []
    centers_list = []
    wh_list = []
    for ind, _ in enumerate(object_prediction_list):
        box = object_prediction_list[ind].bbox.minx, object_prediction_list[ind].bbox.miny, \
            object_prediction_list[ind].bbox.maxx, object_prediction_list[ind].bbox.maxy
        clss = object_prediction_list[ind].category.name
        boxes_list.append(box)
        clss_list.append(clss)
        x1, y1, x2, y2 = box
        center = ( (x1 + x2) / 2, (y1 + y2) / 2)
        centers_list.append(center)
        w,h = abs(x2-x1), abs(y2-y1)
        wh_list.append((w,h, h/w))

    
    centers_array = np.array(centers_list)
    

    # original_image = fit_lines(img=original_image, data=centers_array, slope_threshold=0.5 )
    # cv2.line(original_image, pt1=(int(line_X[0]), int(line_y_ransac[0])), pt2=(int(line_X[-1]), int(line_y_ransac[-1])),thickness=2, color=(170, 255, 0))

    cropped_images_path = "data/cropped_"+os.path.split(input_image_path)[1].split('.')[0]
    if not os.path.exists(cropped_images_path): os.mkdir(cropped_images_path)

    crop_count = 0
    for box, cls, in zip(boxes_list, clss_list):
        x1, y1, x2, y2 = box
        if cls=="Window":
            cropped_image = original_image[int(y1):int(y2), int(x1):int(x2)]
            cropped_file_name = str(crop_count)+'.jpg'
            cv2.imwrite(os.path.join(cropped_images_path,cropped_file_name),cropped_image)
            crop_count+=1
            
    for box, cls, center, whr in zip(boxes_list, clss_list, centers_list, wh_list):
        x1, y1, x2, y2 = box
        if cls=="Window":

            cv2.circle(original_image, center=(int(center[0]), int(center[1])),radius=3,color=(170, 255, 0), thickness=-1)

            cv2.rectangle(original_image, (int(x1), int(y1)), (int(x2), int(y2)), (56, 56, 255), 2)
            
            label = str(round(whr[-1], 4)) # str(cls)
            t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
            cv2.rectangle(original_image, (int(x1), int(y1) - t_size[1] - 3), (int(x1) + t_size[0], int(y1) + 3), (56, 56, 255),
                            -1)
            cv2.putText(original_image,
                        label, (int(x1), int(y1) - 2),
                        0,
                        0.6, [255, 255, 255],
                        thickness=1,
                        lineType=cv2.LINE_AA)
    img_pred_name = input_image_path.split('.')[0]+"_pred_yolo"+".jpg"
    if view_img:
        cv2.imshow("", original_image)
    if save_img:
        cv2.imwrite(img_pred_name, original_image)

   



def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=default_weights_path, help='initial weights path')
    parser.add_argument('--img', type=str, default='data/1.jpg', help='video file path')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-img', action='store_true', help='save results')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    return parser.parse_args()




if __name__ == '__main__':
    opt = parse_opt()
    
    main(input_image_path=opt.img, weights_path=opt.weights, save_img=True)