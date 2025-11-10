import onnxruntime as ort
import numpy as np
from PIL import Image
from typing import Dict, Any, List
import os


class YOLODetectionHandler:
    def __init__(self):
        self.session = None
        self.input_name = None
        self.class_names = self._load_coco_names()

    async def load(self):
        model_path = os.path.join(
            os.path.dirname(__file__), "../../models/yolo11n.onnx"
        )

        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_cpu_mem_arena = True

        self.session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    async def predict(
        self, image: Image.Image, confidence: float = 0.5
    ) -> Dict[str, Any]:
        # Preprocess
        img_array = self._preprocess(image)

        # Inference
        outputs = self.session.run(None, {self.input_name: img_array})

        # Postprocess
        detections = self._postprocess(outputs[0], confidence, image.size)

        return {"detections": detections, "count": len(detections)}

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        image = image.resize((640, 640), Image.BILINEAR)
        image_array = np.array(image).astype(np.float32) / 255.0
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def _postprocess(
        self, output: np.ndarray, confidence: float, original_size: tuple
    ) -> List[Dict]:
        # Output shape: [1, 84, 8400] -> [8400, 84]
        output = output[0].T

        detections = []
        for detection in output:
            scores = detection[4:]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])

            if conf >= confidence:
                x_center, y_center, w, h = detection[:4]

                # Scale to original image size
                scale_x = original_size[0] / 640
                scale_y = original_size[1] / 640

                x1 = (x_center - w / 2) * scale_x
                y1 = (y_center - h / 2) * scale_y
                x2 = (x_center + w / 2) * scale_x
                y2 = (y_center + h / 2) * scale_y

                detections.append(
                    {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": conf,
                        "class_id": class_id,
                        "class_name": self.class_names.get(
                            class_id, f"class_{class_id}"
                        ),
                    }
                )

        return self._apply_nms(detections)

    def _apply_nms(
        self, detections: List[Dict], iou_threshold: float = 0.45
    ) -> List[Dict]:
        if not detections:
            return []

        boxes = np.array([d["bbox"] for d in detections])
        scores = np.array([d["confidence"] for d in detections])

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return [detections[i] for i in keep]

    def _load_coco_names(self) -> Dict[int, str]:
        names = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]
        return {i: name for i, name in enumerate(names)}


class YOLOSegmentationHandler:
    def __init__(self):
        self.session = None
        self.input_name = None
        self.class_names = YOLODetectionHandler()._load_coco_names()

    async def load(self):
        model_path = os.path.join(
            os.path.dirname(__file__), "../../models/yolo11n-seg.onnx"
        )

        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_cpu_mem_arena = True

        self.session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    async def predict(
        self, image: Image.Image, confidence: float = 0.5
    ) -> Dict[str, Any]:
        # Preprocess
        img_array = self._preprocess(image)

        # Inference
        outputs = self.session.run(None, {self.input_name: img_array})

        # Postprocess (simplified - returns detections and raw masks)
        detections = self._postprocess_detections(outputs[0], confidence, image.size)
        masks = outputs[1] if len(outputs) > 1 else None

        return {"detections": detections, "masks": masks, "count": len(detections)}

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        image = image.resize((640, 640), Image.BILINEAR)
        image_array = np.array(image).astype(np.float32) / 255.0
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def _postprocess_detections(
        self, output: np.ndarray, confidence: float, original_size: tuple
    ) -> List[Dict]:
        output = output[0].T

        detections = []
        for detection in output:
            scores = detection[4:84]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])

            if conf >= confidence:
                x_center, y_center, w, h = detection[:4]

                scale_x = original_size[0] / 640
                scale_y = original_size[1] / 640

                x1 = (x_center - w / 2) * scale_x
                y1 = (y_center - h / 2) * scale_y
                x2 = (x_center + w / 2) * scale_x
                y2 = (y_center + h / 2) * scale_y

                detections.append(
                    {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": conf,
                        "class_id": class_id,
                        "class_name": self.class_names.get(
                            class_id, f"class_{class_id}"
                        ),
                    }
                )

        return detections


class YOLOPoseHandler:
    def __init__(self):
        self.session = None
        self.input_name = None
        self.keypoint_names = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]

    async def load(self):
        model_path = os.path.join(
            os.path.dirname(__file__), "../../models/yolo11n-pose.onnx"
        )

        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_cpu_mem_arena = True

        self.session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    async def predict(
        self, image: Image.Image, confidence: float = 0.5
    ) -> Dict[str, Any]:
        # Preprocess
        img_array = self._preprocess(image)

        # Inference
        outputs = self.session.run(None, {self.input_name: img_array})

        # Postprocess
        poses = self._postprocess(outputs[0], confidence, image.size)

        return {"poses": poses, "count": len(poses)}

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        image = image.resize((640, 640), Image.BILINEAR)
        image_array = np.array(image).astype(np.float32) / 255.0
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def _postprocess(
        self, output: np.ndarray, confidence: float, original_size: tuple
    ) -> List[Dict]:
        output = output[0].T

        poses = []
        for idx, detection in enumerate(output):
            conf = float(detection[4])

            if conf >= confidence:
                x_center, y_center, w, h = detection[:4]

                scale_x = original_size[0] / 640
                scale_y = original_size[1] / 640

                x1 = (x_center - w / 2) * scale_x
                y1 = (y_center - h / 2) * scale_y
                x2 = (x_center + w / 2) * scale_x
                y2 = (y_center + h / 2) * scale_y

                # Extract keypoints (starting from index 5)
                keypoints = []
                for i in range(17):
                    kp_x = detection[5 + i * 3] * scale_x
                    kp_y = detection[6 + i * 3] * scale_y
                    kp_conf = float(detection[7 + i * 3])

                    keypoints.append(
                        {
                            "name": self.keypoint_names[i],
                            "x": float(kp_x),
                            "y": float(kp_y),
                            "confidence": kp_conf,
                        }
                    )

                poses.append(
                    {
                        "person_id": idx,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": conf,
                        "keypoints": keypoints,
                    }
                )

        return poses
