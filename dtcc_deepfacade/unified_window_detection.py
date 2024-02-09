import sys, argparse, pathlib, os, json, datetime, logging
from pathlib import Path
import numpy as np
import cv2

logging.basicConfig(level="ERROR")
logger = logging.getLogger("unified_window_detection")

project_folder_path = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_folder_path))

from dtcc_deepfacade.heatmap_fusion_window_detection import heatmap_fusion_detect_windows_single
from dtcc_deepfacade.yolov8_sahi_obj_detection import yolo_detect_windows_single
from dtcc_deepfacade.utils import find_files_in_folder

def compute_intersection_ratio(lt1,rb1, lt2, rb2):
    # intersection ratio between box1 and box2
    # lt: left top coordinate (x,y)
    # rb: right bottom coordinate (x,y)
    xA = max(lt1[0], lt2[0])
    yA = max(lt1[1], lt2[1])
    xB = min(rb1[0], rb2[0])
    yB = min(rb1[1], rb2[1])
    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    current_poly_area = (rb1[0] - lt1[0] + 1) * (lt1[1] - rb1[1] + 1)
    return abs(intersection_area / current_poly_area)


unified_window_predictions = {}
window_count = 0

def is_behind_tree(w_lt,w_rb,trees):

    for (tree_box, score) in trees:
        x1, y1, x2, y2 = tree_box
        lt1 = (x1,y1); rb1 = (x2,y2)
        intersection = compute_intersection_ratio(w_lt, w_rb, lt1, rb1)
        if intersection > 0.2:
            return True
    
    return False

def add_window(hf_pos, box, hf_score, yolo_score,yolo_trees,image_id=None):
    global window_count
    behind_tree = False
    if box is not None:
        x1, y1, x2, y2 = box
        lt = (x1,y1); rb = (x2,y2)
        behind_tree = is_behind_tree(w_lt=lt, w_rb=rb,trees=yolo_trees)
        box = np.array(box,dtype=float).tolist()
    else:
        lt, _, rb, _ = hf_pos
        behind_tree = is_behind_tree(w_lt=lt, w_rb=rb,trees=yolo_trees)
    if hf_pos is not None:
        hf_pos = hf_pos.astype(float).tolist()
    window_preds = {
        "hf":hf_pos, 
        "yolo":box, 
        "hf_score": hf_score if hf_score is None else float(hf_score), 
        "yolo_score":yolo_score if yolo_score is None else float(yolo_score), 
        "behind_tree": behind_tree
    }

    if image_id not in unified_window_predictions.keys():
        unified_window_predictions[image_id] = {}

    unified_window_predictions[image_id][window_count] = window_preds
    window_count += 1


def get_slice_size(image):
    image_h, image_w = image.shape[:2]
    def fit(x):
        if x < 512:
            slice_size = x 
        elif x > 512 and x < 768:
            slice_size = 256
        elif x > 768 and x < 1024:
            slice_size = 256
        else:
            slice_size = 512
        return slice_size

    slice_size = (fit(image_h), fit(image_w))

    slice_image = False
    if slice_size == (image_h,image_w):
        slice_image = True

    return slice_image, slice_size
    

def plot(input_image,input_image_path):

    split_path = os.path.split(input_image_path)
    img_pred_name = split_path[1].split('.')[0]+"_unified_pred_auto_slice"+".jpg"

    preds_folder = os.path.join(split_path[0],"unified_inference")
    os.makedirs(preds_folder, exist_ok=True)
    img_pred_path = os.path.join(preds_folder, img_pred_name)

    ## Plot the windows on the original image.
    for window_id, window in unified_window_predictions[input_image_path].items():
        tree = "T" if window["behind_tree"] else ""
        if window['hf'] is not None:
            pos = np.array(window['hf'])
            pos = pos.astype(np.int32)
            x1 = np.min(pos[:,0]); y1 =np.min(pos[:,1])
            cv2.polylines(input_image, [pos], isClosed=True, color=(170, 255, 0), thickness=2)
            label = tree + str(window_id) + 'hf: ' + str(round(window['hf_score'], 2)) 

        if window['yolo'] is not None:
            x1, y1, x2, y2 = window['yolo']
            cv2.rectangle(input_image, (int(x1), int(y1)), (int(x2), int(y2)), (255,36,0), 2)
            label = tree + str(window_id) + 'y: ' + str(round(window['yolo_score'], 2)) 
    
        t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
        cv2.rectangle(input_image, (int(x1), int(y1) - t_size[1] - 3), (int(x1) + t_size[0], int(y1) + 3), (56, 56, 255), -1)
        cv2.putText(input_image,
                    label, (int(x1), int(y1) - 2),
                    0,
                    0.6, [255, 255, 255],
                    thickness=1,
                    lineType=cv2.LINE_AA)

    cv2.imwrite(img_pred_path, input_image)



def unified_window_detection_single(input_image_path = "data/1.jpg", auto_slice=True, slice_image=True, slice_w=256, slice_h=256,plot_predictions=True, save_predictions=False):
    # Check image path
    if not Path(input_image_path).exists():
        raise FileNotFoundError(f"Source path '{input_image_path}' does not exist.")
    input_image = cv2.imread(input_image_path)
    
    if auto_slice:
        slice_image, (slice_h, slice_w) = get_slice_size(image=input_image)

    heatmap_fusion_windows = heatmap_fusion_detect_windows_single(input_image=input_image, slice=slice_image, slice_h=slice_h,slice_w=slice_w,overlap_ratio=0.6)
    
    yolo_windows, yolo_trees = yolo_detect_windows_single(input_image=input_image,slice_w=slice_w, slice_h=slice_h)

    ## If duplicates exists keep the heatmap fusion version
    hf2yolo_intersections = np.zeros((len(heatmap_fusion_windows), len(yolo_windows)))
    yolo2hf_intersections = np.zeros_like(hf2yolo_intersections)

    ## TODO check the angles of heatmap fusion prediction before finalizing.
    global window_count 
    window_count = 0

    for i,(hf_pos, score) in enumerate(heatmap_fusion_windows):
        lt1, lb1, rb1, rt1 = hf_pos
       
        for j, (box,box_score) in enumerate(yolo_windows):
   
            x1, y1, x2, y2 = box
            lt2 = (x1,y1); rb2 = (x2,y2)
        
            hf2yolo = compute_intersection_ratio(lt1, rb1,lt2, rb2)
            yolo2hf = compute_intersection_ratio(lt2, rb2, lt1, rb1)
            
            hf2yolo_intersections[i,j] = hf2yolo
            yolo2hf_intersections[i,j] = yolo2hf
            if hf2yolo > 0.1 or yolo2hf > 0.1:
                add_window(hf_pos,box, score,box_score,yolo_trees,image_id=input_image_path)
            
    
    hf_no_intersection_pos = np.where(np.sum(hf2yolo_intersections,axis=1)<0.1)

    yolo_no_intersection_pos = np.where(np.sum(yolo2hf_intersections,axis=0)<0.1)

    for i in hf_no_intersection_pos[0]:
        hf_pos, score = heatmap_fusion_windows[i]
        add_window(hf_pos,None,score,None,yolo_trees,image_id=input_image_path)

    for i in yolo_no_intersection_pos[0]:
        box, score = yolo_windows[i]
        add_window(None,box,None,score,yolo_trees,image_id=input_image_path)

    
    # # print(f"slice h: {slice_h}, w: {slice_w}")
    

    if plot_predictions:
        plot(input_image=input_image,input_image_path=input_image_path)

    if save_predictions:
        save_results()

    return unified_window_predictions

    

def unified_window_detection_multiple(images_directory_path = "data",auto_slice=True, slice_image=True, slice_w=512, slice_h=256, save_predictions=True, plot_predictions=True):
    # Check path
    if not Path(images_directory_path).exists():
        raise FileNotFoundError(f"Source path '{images_directory_path}' does not exist.")
    
    image_paths = find_files_in_folder(images_directory_path, extension=('.jpg','.png'))
    

    for i, input_image_path in enumerate(image_paths):
        try:
            print(f"[{i}]", input_image_path)
            unified_window_detection_single(input_image_path=input_image_path,auto_slice=auto_slice, slice_image=slice_image, slice_w=slice_w, slice_h=slice_h, save_predictions=False, plot_predictions=plot_predictions)
        except BaseException as e:
            logger.exception(f"################### failed prediction: {input_image_path}")
    
    if save_predictions:
        save_results()

    return unified_window_predictions


def save_results():
    cwd = os.getcwd()
    results_path = os.path.join(cwd, "results")
    os.makedirs(results_path, exist_ok=True)
    predictions_save_path = os.path.join(results_path, f"unified_inference_{datetime.datetime.now().isoformat()}.json")
    json.dump(unified_window_predictions, open(predictions_save_path,'w'), indent=4)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data/examples', help='image file / folder path')
    parser.add_argument('--plot', action='store_true', help='plot results')
    parser.add_argument('--save_results', action='store_true', help='save window detction results in json format')

    args =  parser.parse_args()

    path = args.path
    if os.path.exists(path):
        if os.path.isfile(path):
            unified_window_detection_single(input_image_path=args.path, auto_slice=True, slice_w=512, slice_h=256)
        else:
            unified_window_detection_multiple(images_directory_path=path)


    