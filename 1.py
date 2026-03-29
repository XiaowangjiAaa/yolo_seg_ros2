from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")
model.export(format="ncnn", imgsz=320)