from ultralytics import YOLO

if __name__ == '__main__':
    # Load the YOLO model
    model = YOLO('yolov8m.yaml')  # Specify the version you want to use (e.g., yolov8s.yaml for small model)

    # Train the model
    model.train(data="D:\\project-2\\Completely_new_datasets\\My First Project.v4i.yolov8\\data.yaml", epochs=100, batch=8, imgsz=640)

    # Save the trained model
    model.save('D:\\project-2\\Completely_new_datasets\\My_First_Project.v4i.yolov8\\best1.pt')
    