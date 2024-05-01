import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import random
from segment_anything import sam_model_registry, SamPredictor
from tqdm.autonotebook import tqdm
from scipy.interpolate import interp1d


def find_pupil(img):
    img = cv2.medianBlur(img, 15)
    img = cv2.Canny(img, 0, 50)
    param1 = 200  # 200
    param2 = 120  # 150
    decrement = 1
    circles = None
    while circles is None and param2 > 20:
        # HoughCircles
        circles = cv2.HoughCircles(
            img,
            cv2.HOUGH_GRADIENT,
            1,
            1,
            param1=param1,
            param2=param2,
            minRadius=20,
            maxRadius=80,
        )

        if circles is not None:
            break

        param2 -= decrement

    if circles is None:
        return None, None, None

    return circles.astype(int)[0][0]


def normalization(img, pupil_circle, iris_circle, M=16, N=400, offset=0):

    normalized = np.zeros((M, N))

    for i in range(N):
        begin = pupil_circle
        end = iris_circle

        xspace = np.linspace(begin[i][0], end[i][0], M)
        yspace = np.linspace(begin[i][1], end[i][1], M)
        normalized[:, i] = [
            (
                img[int(y), int(x)]
                if 0 <= int(x) < img.shape[1] and 0 <= int(y) < img.shape[0]
                else 0
            )
            for x, y in zip(xspace, yspace)
        ]
    return normalized


def find_iris(predictor, image_path):
    image_og = cv2.imread(image_path)
    if image_og is None:
        print(f"Image not found: {image_path}")
        return None, None, None, None
    image_og = cv2.cvtColor(image_og, cv2.COLOR_BGR2RGB)
    _, thresh = cv2.threshold(image_og, 120, 255, cv2.THRESH_BINARY)
    filter_img = cv2.cvtColor(image_og, cv2.COLOR_RGB2GRAY)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_RGB2GRAY)
    thresh_mask = thresh != 0
    filter_img[thresh_mask] = np.mean(filter_img)
    filter_img = cv2.medianBlur(filter_img, 13)

    predictor.set_image(image_og)

    pupil_circle = find_pupil(filter_img)
    if any(element is None for element in pupil_circle):
        return None, None, None, None
    input_point = np.array(
        [
            [pupil_circle[0], pupil_circle[1]],
            [pupil_circle[0], pupil_circle[1] - 50],
            [pupil_circle[0], pupil_circle[1] + 70],
            [pupil_circle[0] - 70, pupil_circle[1]],
            [pupil_circle[0] + 70, pupil_circle[1]],
        ]
    )
    input_label = np.array([1, 0, 0, 0, 0])
    masks_pupil, scores_pupil, logits_pupil = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    input_point = np.array(
        [
            [pupil_circle[0], pupil_circle[1]],
            [pupil_circle[0], pupil_circle[1] - 50],
            [pupil_circle[0], pupil_circle[1] + 70],
            [pupil_circle[0] - 70, pupil_circle[1]],
            [pupil_circle[0] + 70, pupil_circle[1]],
            [pupil_circle[0] - 120, pupil_circle[1]],
            [pupil_circle[0] + 120, pupil_circle[1]],
            [pupil_circle[0] - 120, pupil_circle[1] + 40],
            [pupil_circle[0] + 120, pupil_circle[1] + 40],
            [pupil_circle[0] - 120, pupil_circle[1] - 40],
            [pupil_circle[0] + 120, pupil_circle[1] - 40],
        ]
    )
    input_label = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    mask_input = logits_pupil[np.argmax(scores_pupil), :, :]
    masks_iris, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )

    return masks_pupil, masks_iris, input_point, input_label


def get_outline(mask):
    # Convert the mask to uint8 for compatibility with OpenCV
    mask = cv2.medianBlur(mask.astype(np.uint8) * 255, 13)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return None

    # Extract the coordinates of the contours
    outline_coords = []
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the coordinates of the largest contour
    outline_coords = largest_contour.squeeze().tolist()

    return np.array(outline_coords)


def get_inter(outline, N=400):
    if outline is None:
        return None

    inter_x = interp1d(np.arange(0, len(outline[:, 0]), 1), outline[:, 0])
    ynew_x = np.arange(0, len(outline[:, 0]) - 1, (len(outline[:, 0]) - 1) / N)
    xnew_x = inter_x(ynew_x)

    inter_y = interp1d(np.arange(0, len(outline[:, 0]), 1), outline[:, 1])
    xnew_y = np.arange(0, len(outline[:, 0]) - 1, (len(outline[:, 0]) - 1) / N)
    ynew_y = inter_y(xnew_y)

    interpolated_coords = np.column_stack((xnew_x, ynew_y))
    return interpolated_coords


def load_predictor(model_type):
    device = "cuda"
    model_paths = {
        "vit_b": "D:/Users/jimyj/Desktop/TAIST/Thesis/Source_Code/main/module/segment_anything/checkpoints/sam_vit_b_01ec64.pth",
        "vit_l": "D:/Users/jimyj/Desktop/TAIST/Thesis/Source_Code/main/module/segment_anything/checkpoints/sam_vit_l_0b3195.pth",
        "vit_h": "D:/Users/jimyj/Desktop/TAIST/Thesis/Source_Code/main/module/segment_anything/checkpoints/sam_vit_h_4b8939.pth",
    }

    sam = sam_model_registry[model_type](checkpoint=model_paths[model_type])
    sam.to(device=device)

    predictor = SamPredictor(sam)
    return predictor


def find_iris_box(predictor, image_path):
    image_og = cv2.imread(image_path)
    if image_og is None:
        print(f"Image not found: {image_path}")
        return [], [], [], []
    image_og = cv2.cvtColor(image_og, cv2.COLOR_BGR2RGB)
    _, thresh = cv2.threshold(image_og, 120, 255, cv2.THRESH_BINARY)
    filter_img = cv2.cvtColor(image_og, cv2.COLOR_RGB2GRAY)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_RGB2GRAY)
    thresh_mask = thresh != 0
    filter_img[thresh_mask] = np.mean(filter_img)
    filter_img = cv2.medianBlur(filter_img, 17)

    predictor.set_image(image_og)

    pupil_circle = find_pupil(filter_img)
    if any(element is None for element in pupil_circle):
        return [], [], [], []
    input_point = np.array(
        [
            [pupil_circle[0], pupil_circle[1]],
            [pupil_circle[0], pupil_circle[1] - 50],
            [pupil_circle[0], pupil_circle[1] + 70],
            [pupil_circle[0] - 70, pupil_circle[1]],
            [pupil_circle[0] + 70, pupil_circle[1]],
        ]
    )
    input_label = np.array([1, 0, 0, 0, 0])
    masks_pupil, scores_pupil, logits_pupil = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    input_point = np.array([[pupil_circle[0], pupil_circle[1]]])
    input_box = np.array(
        [
            pupil_circle[0] - 120,
            pupil_circle[1] - 100,
            pupil_circle[0] + 120,
            pupil_circle[1] + 120,
        ]
    )
    input_label = np.array([1])
    mask_input = logits_pupil[np.argmax(scores_pupil), :, :]
    masks_iris, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box[None, :],
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )

    return masks_pupil, masks_iris, input_point, input_box, input_label
