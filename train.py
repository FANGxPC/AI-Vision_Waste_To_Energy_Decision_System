from ultralytics import YOLO

def main():
    # 1. Load the pre-trained 'Nano' model
    model = YOLO('yolov8n.pt') 

    # 2. Start training
    # We use the relative path to your dataset folder
    model.train(
        data='Trash Detection.v1i.yolov8/data.yaml', 
        epochs=30, 
        imgsz=640,
        plots=True
    )

if __name__ == '__main__':
    main()
