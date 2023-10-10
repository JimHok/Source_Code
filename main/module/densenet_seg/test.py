import torch
from module.densenet_seg.dataset import transform
from module.densenet_seg.models import model_dict
from module.densenet_seg.utils import get_predictions
import numpy as np
import matplotlib.pyplot as plt


def run_prediction(image_path, model_name, model_path, use_gpu=True, plot=False):
    device = torch.device("cuda" if use_gpu else "cpu")
    model = model_dict[model_name].to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    img = plt.imread(image_path)
    img = transform(img)
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        predict = get_predictions(output)
        pred_img = predict[0].cpu().numpy()/3.0
        inp = img.squeeze().cpu().numpy() * 0.5 + 0.5
        img_orig = np.clip(inp, 0, 1)
        img_orig = np.array(img_orig)
        if plot:
            combine = np.hstack([img_orig, pred_img])
            plt.imshow(combine)
            plt.show()

    return img_orig, pred_img
