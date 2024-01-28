from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from typing_extensions import Annotated

app = FastAPI()
model = YOLO("best.pt")


def predict(image):
    results = model(image)
    image = np.array(image)
    return results, image


def to_list(results):
    return results[0].boxes.cpu().xyxy.tolist()


def overlay_rectangles(image, xyxys):
    for xyxy in xyxys:
        cv2.rectangle(
            image,
            (int(xyxy[0]), int(xyxy[1])),
            (int(xyxy[2]), int(xyxy[3])),
            (0, 255, 0),
            3,
        )
    image = Image.fromarray(image)
    return image


results, image = predict(Image.open("test2.jpg"))
results = to_list(results)
print("done")
(overlay_rectangles(image, results))
print("rec")


@app.post("/inference/")
async def inference(file: UploadFile):
    print({"file": file.filename})
    results, image = predict(Image.open("test2.jpg"))
    return to_list(results)


@app.get("/")
async def resp():
    return {"message": "server up"}
