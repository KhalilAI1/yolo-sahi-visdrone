from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO


detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=r"runs\detect\yolov8n_visdrone-4\weights\best.pt",
    confidence_threshold=0.3,
    device="cuda"
)

image_path = r"visdrone_subset\images\val\0000249_01635_d_0000006.jpg"

result = get_sliced_prediction(
    image_path,
    detection_model,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    postprocess_type="NMS",
    postprocess_match_threshold=0.5,
)

result.export_visuals(export_dir="test_sahi_inference_nms", hide_conf=True)
print("Normal inference done ✓")