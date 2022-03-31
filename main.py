### Main Code Imports
import multiprocessing as mp
from multiprocessing import Process, Queue

import cv2
import os
import numpy as np
import torch, torchvision
import detectron2

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances


### API Imports
import io
import json
import base64
import uuid
import logging
import sys
import uvicorn
import nest_asyncio
from typing import List
from PIL import Image
from io import BytesIO
from starlette.responses import Response, StreamingResponse, JSONResponse, HTMLResponse
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect



###This is main code
DatasetCatalog.clear()
register_coco_instances("road_train", {}, "/content/gdrive/MyDrive/Dataset/New Road Train/json_annotation_train.json", "/content/gdrive/MyDrive/Dataset/New Road Train")
MetadataCatalog.get("road_train").set(thing_classes=["Road"])

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def DrawLinesP(frame,linesP):
    frame_LinesPCanvas = np.zeros_like(frame)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(frame_LinesPCanvas, (l[0], l[1]), (l[2], l[3]), (255,0,0), 3, cv2.LINE_AA)
     
            
    frame_OutputP = cv2.addWeighted(frame, 0.9, frame_LinesPCanvas, 1, 1)
    return frame_OutputP

setup_logger()

# Log to stdout
logging.basicConfig(
    stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s"
)

# Number of processors
logging.info(f"Number of processors: { mp.cpu_count() } ")

# GPU is available
gpu = torch.cuda.is_available()
logging.info(f"GPU available - { gpu }")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = os.path.join("Model/roadDetection_Final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1


if not gpu:
    cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)


def detectron2_predict(frame):
        #cv2_imshow(frame)
        outputs = predictor(frame)
        # Get pred_boxes from Detectron2 prediction outputs
        boxes = outputs["instances"].pred_boxes

        # Select 1 bounding box:
        try:
              box = list(boxes)[0].detach().cpu().numpy()
              x_1 = box[0]
              y_1 = box[1]
              x_2 = box[2]
              y_2 = box[3]
              # Crop the PIL image using predicted box coordinates
                  
              contours = np.array([[int(x_1),int(y_1)], [int(x_2),int(y_1)], [int(x_2),int(y_2)], [int(x_1),int(y_2)]])
              image = np.zeros((frame.shape)).astype(frame.dtype)
              frame_poly = cv2.fillPoly(image, pts = [contours], color =(255,255,255))
              frame_bit = cv2.bitwise_and(frame,frame_poly)
              frame_HSV = cv2.cvtColor(frame_bit , cv2.COLOR_BGR2HSV)
              frame_Canny = cv2.Canny(frame_HSV,50,150,apertureSize = 3)
              frame_HoughP = cv2.HoughLinesP(frame_Canny, 1, np.pi / 180, 50, None, 100, 5)
              frame_LinesP = DrawLinesP(frame,frame_HoughP)

              v = Visualizer(frame_LinesP[:, :, ::-1], MetadataCatalog.get("road_train"),scale=0.8, instance_mode=ColorMode.SEGMENTATION)
              vDrawed = v.draw_instance_predictions(outputs["instances"].to("cpu"))
              return(vDrawed.get_image()[:, :, ::-1])
        except:
              print("Error: No object has been found.")
              return JSONResponse(status_code=201, content={"message": "Error: No object has been found."})
        
              
        #cv2.imshow(vDrawed.get_image()[:, :, ::-1])
        #cv2.imwrite("/content/gdrive/MyDrive/Run Result/vDrawed%d.jpg" % imgCount, vDrawed.get_image()[:, :, ::-1])
       
###Below is API
# FastAPI
app = FastAPI(
    title="Detectron2 Server API",
    description="""Visit port 8088/docs for documentation.""",
    version="0.0.1",
)


class ConnectionManager:
    """Web socket connection manager."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


conn_mgr = ConnectionManager()


def base64_encode_img(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    buffered.seek(0)
    img_byte = buffered.getvalue()
    encoded_img = "data:image/png;base64," + base64.b64encode(img_byte).decode()
    return encoded_img


@app.get("/")
def home():
    return {"message": "This is a home page."}

@app.post("/detectron2")    
async def process_detectron2(file: UploadFile = File(...)):
    contents = file.file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    converted_img = detectron2_predict(np.array(img)) 
    
    
    try:    
      img2 = Image.fromarray(converted_img.astype('uint8'), 'RGB')
      image_byte_arr = io.BytesIO()
      img2.save(image_byte_arr, format='PNG')
      image_byte_arr = image_byte_arr.getvalue()
      return StreamingResponse(io.BytesIO(image_byte_arr), media_type='image/png')
    except:
      print("Error: No object has been found.")
      return JSONResponse(status_code=202, content={"message": "Error: Error: No object has been detected by Detectron2"})

@app.websocket("/detectron2_ws/{client_id}")
async def process_detectron2_ws(websocket: WebSocket, client_id: int):
    await conn_mgr.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()

            # Convert to PIL image
            image = data[data.find(",") + 1 :]
            dec = base64.b64decode(image + "===")
            image = Image.open(BytesIO(dec)).convert("RGB")

            # Process the image
            name = f"/data/{str(uuid.uuid4())}.png"
            image.filename = name
            converted_image = detectron2_predict(np.array(image))

            result = {
                "output": base64_encode_img(converted_image),
            }
            # logging.info("-----", json.dumps(result))

            # Send back the result
            await conn_mgr.send_message(json.dumps(result), websocket)

            # await conn_mgr.broadcast(f"Client #{client_id} says: {data}")
    except WebSocketDisconnect:
        conn_mgr.disconnect(websocket)
        await conn_mgr.broadcast(f"Client #{client_id} left the chat")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8088, reload=True)
