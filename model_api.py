import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify
from models.common import DetectMultiBackend
from utils.datasets import LoadImagesForApi
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import save_one_box_for_api, crop_box
from utils.torch_utils import select_device
from data.dataset import match_template



app = Flask(__name__)
conf_thres = 0.7
iou_thres = 0.45
weights = "./runs/train/clahe/exp2/weights/best.pt"
device = select_device("0")
model = DetectMultiBackend(weights, device=device)




@app.route("/predict", methods=["post"])
def predict():
    img = request.files["image"]
    block = request.files["block"]
    block_array = np.frombuffer(block.read(), np.uint8)
    block_img = cv2.imdecode(block_array, cv2.IMREAD_COLOR)
    req_img_name = request.values.get("name")

    dataset = LoadImagesForApi(img, req_img_name)
    im1, im2, img_src = dataset.__next__()
    imc = img_src.copy()
    im1 = torch.from_numpy(im1).to(device)

    im1 = im1.float()
    im1 /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im1.shape) == 3:
        im1 = im1[None]  # expand for batch dim

    pred = model(im1)
    pred = non_max_suppression(pred, conf_thres, iou_thres, None, True, max_det=1000)

    boxs = []

    for i, det in enumerate(pred):
        det[:, :4] = scale_coords(im1.shape[2:], det[:, :4], imc.shape).round()
        for *xyxy, conf, cls in reversed(det):
            save_one_box_for_api(xyxy, imc, file=f'data/dataset/temp/crop/{req_img_name}_{i}_crop.jpg', BGR=True)
            xyxy_temp = xyxy.copy()
            crop = crop_box(xyxy_temp, imc, BGR=True)
            xyxy_temp = torch.tensor(xyxy_temp).view(-1, 4)
            b = xyxy2xywh(xyxy_temp)  # boxes
            box = {
                "xy": (b[:, 0].cpu().numpy().tolist()[0], b[:, 1].cpu().numpy().tolist()[0]),
                "similar": match_template(crop, block_img),
                "xyxy": [i.cpu().numpy().tolist() for i in xyxy],
                "conf": conf.cpu().numpy().tolist(),
                "cls": cls.cpu().numpy().tolist()
            }
            boxs.append(box)
    del dataset
    return jsonify(boxs)


if __name__ == '__main__':
    app.run(host="192.168.0.103", port=8090, debug=True)
