"""
Object Detection Module - YOLO Wrapper Class

WHY THIS DESIGN:
1. Class-based: Can create multiple detector instances (one per camera)
2. Model caching: Load model ONCE at startup, not every frame
3. Returns structured data: DetectionResult objects (not None)
4. Supports multi-camera: Each detector independent
5. Extensible: Easy to add vehicle tracking, speed estimation

PERFORMANCE NOTE:
- Original code loaded YOLO model EVERY FRAME = very slow
- This code loads it ONCE in __init__ = 100x faster
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from ultralytics import YOLO
from loguru import logger


@dataclass
class Detection:
    """Represents a single detected object"""
    class_id: int
    class_name: str
    confidence: float
    x1: int  # Top-left x
    y1: int  # Top-left y
    x2: int  # Bottom-right x
    y2: int  # Bottom-right y

    @property
    def center(self) -> Tuple[int, int]:
        """Center point of bounding box"""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def width(self) -> int:
        """Width of bounding box"""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Height of bounding box"""
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """Area of bounding box"""
        return self.width * self.height


@dataclass
class DetectionResult:
    """Result from a single frame detection"""
    frame_id: int
    timestamp: float
    camera_id: str
    detections: List[Detection]
    vehicle_count: int
    pedestrian_count: int
    total_count: int
    frame: Optional[np.ndarray] = None  # Can store frame for visualization

    def get_vehicles(self) -> List[Detection]:
        """Get only vehicle detections"""
        return [d for d in self.detections if d.class_name in ['car', 'truck', 'bus', 'motorcycle']]

    def get_pedestrians(self) -> List[Detection]:
        """Get only pedestrian detections"""
        return [d for d in self.detections if d.class_name == 'person']


class ObjectDetector:
    """
    YOLO Object Detector

    Responsibilities:
    - Load and cache YOLO model
    - Detect objects in frames
    - Return structured detection results
    - Optionally visualize detections
    """

    # COCO dataset class mappings (YOLO trained on COCO)
    VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}
    PEDESTRIAN_CLASSES = {'person'}

    def __init__(self,
                 model_weights: str,
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 device: str = 'cpu',
                 camera_id: str = 'default'):
        """
        Initialize detector

        Args:
            model_weights: Path to YOLOv8 weights (e.g., 'yolov8n.pt')
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS (Non-Maximum Suppression)
            device: 'cpu' or 'cuda' (GPU)
            camera_id: Identifier for this camera
        """
        self.model_weights = model_weights
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.camera_id = camera_id
        self.frame_count = 0

        logger.info(f"Loading YOLO model: {model_weights} on {device}")

        try:
            # Load model ONCE at initialization
            self.model = YOLO(model_weights)
            self.model.to(device)
            logger.info(f"✓ Model loaded successfully for camera {camera_id}")
        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            raise

    def detect(self, frame: np.ndarray, return_frame: bool = False) -> DetectionResult:
        """
        Detect objects in a frame

        Args:
            frame: Input frame (numpy array)
            return_frame: If True, include annotated frame in result

        Returns:
            DetectionResult object with all detections and counts
        """
        self.frame_count += 1

        try:
            # Run inference
            # conf: confidence threshold
            # iou: IoU threshold for NMS
            results = self.model(frame,
                               conf=self.confidence_threshold,
                               iou=self.iou_threshold,
                               verbose=False)

            detections = []
            vehicle_count = 0
            pedestrian_count = 0

            # Process results
            for result in results:
                if result.boxes is not None:
                    # Get class names from model
                    class_names = result.names

                    # Extract detections
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                        confidence = float(box.conf[0])  # Confidence score
                        class_id = int(box.cls[0])  # Class ID
                        class_name = class_names[class_id]  # Class name (e.g., 'car')

                        # Create Detection object
                        detection = Detection(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=confidence,
                            x1=x1, y1=y1, x2=x2, y2=y2
                        )

                        detections.append(detection)

                        # Count vehicles and pedestrians
                        if class_name in self.VEHICLE_CLASSES:
                            vehicle_count += 1
                        elif class_name in self.PEDESTRIAN_CLASSES:
                            pedestrian_count += 1

            # Create result object
            result_obj = DetectionResult(
                frame_id=self.frame_count,
                timestamp=cv2.getTickCount() / cv2.getTickFrequency(),
                camera_id=self.camera_id,
                detections=detections,
                vehicle_count=vehicle_count,
                pedestrian_count=pedestrian_count,
                total_count=len(detections),
                frame=frame if return_frame else None
            )

            logger.debug(f"[{self.camera_id}] Detected: {vehicle_count} vehicles, {pedestrian_count} pedestrians")

            return result_obj

        except Exception as e:
            logger.error(f"Error during detection: {e}")
            # Return empty result on error
            return DetectionResult(
                frame_id=self.frame_count,
                timestamp=cv2.getTickCount() / cv2.getTickFrequency(),
                camera_id=self.camera_id,
                detections=[],
                vehicle_count=0,
                pedestrian_count=0,
                total_count=0,
                frame=None
            )

    def detect_with_visualization(self, frame: np.ndarray) -> Tuple[DetectionResult, np.ndarray]:
        """
        Detect objects and return annotated frame

        Returns:
            (DetectionResult, annotated_frame)
        """
        result = self.detect(frame, return_frame=True)
        annotated_frame = self._annotate_frame(frame, result)
        return result, annotated_frame

    def _annotate_frame(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        annotated = frame.copy()

        for detection in result.detections:
            # Choose color based on class
            if detection.class_name in self.VEHICLE_CLASSES:
                color = (0, 255, 0)  # Green for vehicles
            elif detection.class_name in self.PEDESTRIAN_CLASSES:
                color = (0, 0, 255)  # Red for pedestrians
            else:
                color = (255, 255, 0)  # Cyan for others

            # Draw bounding box
            cv2.rectangle(annotated,
                         (detection.x1, detection.y1),
                         (detection.x2, detection.y2),
                         color, 2)

            # Draw label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            cv2.putText(annotated, label,
                       (detection.x1, detection.y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, color, 2)

        # Draw counts at top
        info = f"Vehicles: {result.vehicle_count} | Pedestrians: {result.pedestrian_count}"
        cv2.putText(annotated, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return annotated

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'model'):
            del self.model
        logger.info(f"Detector for camera {self.camera_id} cleaned up")


if __name__ == '__main__':
    """Quick test of detector"""
    # Test with webcam
    detector = ObjectDetector(
        model_weights='yolov8n.pt',
        device='cpu',
        camera_id='test_camera'
    )

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect with visualization
        result, annotated = detector.detect_with_visualization(frame)

        print(f"Frame {result.frame_id}: {result.vehicle_count} vehicles, {result.pedestrian_count} pedestrians")

        cv2.imshow('Detection', annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.cleanup()
