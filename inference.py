import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json

checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

def show_mask(mask, ax, random_color=False, borders = True):
    # 显示遮罩
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    # 显示提示点：前景点为绿色，背景为红色
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    # 显示坐标框
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, name="default", point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        if i == 0:
            plt.savefig(f"new_best_output/{name}.jpg", bbox_inches='tight')
        plt.savefig(f"new_output/{name}_{i+1}.jpg", bbox_inches='tight')
        plt.close() 

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    folder = "final_labeled"
    file_names = os.listdir(folder)
    for file_name in file_names:
        if file_name.endswith('.json'):
            with open(os.path.join(folder, file_name), 'r', encoding='utf-8') as file:
                data = json.load(file)
            image_name = file_name.replace('.json', '.jpg')
            shapes = data['shapes']
            for shape in shapes:
                if shape['label'] == 'KELOID_BODY':
                    points = shape['points']
                    array = np.array(points)
                    break
            assert array is not None, 'No KELOID_BODY found in {}'.format(file_name)
            image_path = os.path.join(folder, image_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)
            labels = np.ones((array.shape[0],))
            masks, scores, logits = predictor.predict(point_coords=array, point_labels=labels)
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]
            print(f"finish predict {image_name}")
            image_name = image_name.split(".")[0]
            show_masks(image, masks, scores, name=image_name)