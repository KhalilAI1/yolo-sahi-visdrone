from ultralytics import YOLO
from sahi import AutoDetectionModel




if __name__ == '__main__':
    detection_model = AutoDetectionModel.from_pretrained(
                            model_type="yolov8",
                            model_path=r"runs\\detect\\yolov8n_visdrone-4\\weights\\best.pt",
                            confidence_threshold=0.3,
                            device="cuda"
                        )
