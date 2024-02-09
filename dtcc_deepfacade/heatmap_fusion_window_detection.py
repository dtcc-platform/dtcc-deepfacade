import cv2, pathlib, sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

project_folder_path = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_folder_path))

from dtcc_deepfacade.win_det_heatmaps.infer import get_windows_with_score

def sliding_window(image, overlap_ratio=0.5, window_size = (512,512), ):
    # step_size: percentage of height and width
    # get the window and image sizes
    h, w = window_size
    image_h, image_w = image.shape[:2]
    y_step_size = int(h * (1 - overlap_ratio))
    x_step_size = int(w * (1 - overlap_ratio))

    # loop over the image, taking steps of size `step_size`
    for y in range(0, image_h, y_step_size):
        for x in range(0, image_w, x_step_size):
            # define the window
            window = image[y:y + h, x:x + w]
            # if the window is below the minimum window size, ignore it
            if window.shape[:2] != window_size:
                continue
            # yield the current window
            yield (x, y, window)


def compute_intersection_ratio(current, other):
    lt1, lb1, rb1, rt1 = current
    lt2, lb2, rb2, rt2 = other

    xA = max(lt1[0], lt2[0])
    yA = max(lt1[1], lt2[1])
    xB = min(rb1[0], rb2[0])
    yB = min(rb1[1], rb2[1])
    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    current_poly_area = (rb1[0] - lt1[0] + 1) * (lt1[1] - rb1[1] + 1)
    return abs(intersection_area / current_poly_area)

def identify_duplicate_predictions(window_positions_with_score):
    n_preds = len(window_positions_with_score)
    elements = {}
    for i in range(n_preds):
        current_pos = window_positions_with_score[i][0]
        for j in range(n_preds):
            if i != j:
                other_pos = window_positions_with_score[j][0]
                ratio = compute_intersection_ratio(current_pos, other_pos)
                if ratio > 0.1:
                    elements[f"{i},{j}"] = ratio

    remove_set = set()
    for k,v in elements.items():
        w1,w2 = k.split(',')
        
        if elements[f"{w2},{w1}"] < v:
            remove_item = w1
        else:
            remove_item = w2
        
        remove_set.add(int(remove_item))

    ignore_list = list(remove_set)

    return ignore_list

def plot_predictions(image, pos,score):
    pos = pos.astype(np.int32)
    x1 = np.min(pos[:,0]); y1 =np.min(pos[:,1])
    cv2.polylines(image, [pos], isClosed=True, color=(56, 56, 255), thickness=2)
    label = str(round(score, 4)) 
    t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
    cv2.rectangle(image, (int(x1), int(y1) - t_size[1] - 3), (int(x1) + t_size[0], int(y1) + 3), (56, 56, 255),
                    -1)
    cv2.putText(image,
                label, (int(x1), int(y1) - 2),
                0,
                0.6, [255, 255, 255],
                thickness=1,
                lineType=cv2.LINE_AA)


def heatmap_fusion_detect_windows_single(input_image, slice=True, slice_w=256, slice_h=256,overlap_ratio=0.6):
    image_h, image_w = input_image.shape[:2]

    if slice:

        segments = [ seg for seg in sliding_window(input_image, overlap_ratio=overlap_ratio, window_size=(slice_w, slice_h))]

        offset_windows_with_score = get_windows_with_score(input_image_list=segments)

        window_positions_with_score = []
        for win_pred, x_offset, y_offset in offset_windows_with_score:
            for win in win_pred:
                pos = win['position'][:,:2] + np.array([x_offset, y_offset])
                
                window_positions_with_score.append((pos,win['score']))
    else:
        offset_windows_with_score = get_windows_with_score(input_image_list= [(image_w, image_h, input_image)])

        window_positions_with_score = []
        for win_pred, x_offset, y_offset in offset_windows_with_score:
            for win in win_pred:
                pos = win['position'][:,:2]
                
                window_positions_with_score.append((pos,win['score']))

    ignore_list = identify_duplicate_predictions(window_positions_with_score)

    return [ v for i, v in enumerate(window_positions_with_score) if i not in ignore_list]


def main():
    input_image_path = "data/1.jpg"
    image = cv2.imread(input_image_path)
    w, h = 512, 256

    segments = [ seg for seg in sliding_window(image, overlap_ratio=0.6, window_size=(w, h))]

    offset_windows_with_score = get_windows_with_score(image_segments=segments)

    window_positions_with_score = []
    for win_pred, x_offset, y_offset in offset_windows_with_score:
        for win in win_pred:
            pos = win['position'][:,:2] + np.array([x_offset, y_offset])
            
            window_positions_with_score.append((pos,win['score']))

    ignore_list = identify_duplicate_predictions(window_positions_with_score)

    for i, (pos, score) in enumerate(window_positions_with_score):
        if i not in ignore_list:
            plot_predictions(image=image,pos=pos,score=score)
            
    img_pred_name = input_image_path.split('.')[0]+"_pred"+".jpg"
    cv2.imwrite(img_pred_name, image)



if __name__=='__main__':
    main()
