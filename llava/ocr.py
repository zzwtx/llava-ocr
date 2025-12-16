import sys
# print("Starting OCR script...", file=sys.stderr)
import os
import cv2
import numpy as np
import os as _os
# Ensure CUDA/cuDNN libs shipped inside the conda env are visible to the runtime before importing onnxruntime.
# ONNXRuntime CUDA provider may require libcudnn.so.9 in LD_LIBRARY_PATH. Add common conda-installed locations if present.
_cudnn_paths = [
    "/home/xsf/miniconda3/envs/tllm0120/lib/python3.10/site-packages/nvidia/cudnn/lib",
    "/home/xsf/miniconda3/envs/tllm0120/lib/",
]
_existing = _os.environ.get("LD_LIBRARY_PATH", "")
for _p in _cudnn_paths:
    if _p and _p not in _existing and _os.path.isdir(_p):
        _existing = _p + (":" + _existing if _existing else "")
        _os.environ["LD_LIBRARY_PATH"] = _existing
try:
    import onnxruntime as ort
except Exception:
    # Fall back to import without CUDA provider; session creation will warn if CUDA provider unavailable.
    import onnxruntime as ort
import math
from shapely.geometry import Polygon
import pyclipper

class OCRSystem:
    def __init__(self, model_dir, keys_path):
        self.model_dir = model_dir
        self.keys_path = keys_path
        self.keys = self.load_keys(keys_path)
        
        # Determine device based on MPI rank if available, otherwise default
        try:
            from tensorrt_llm._utils import mpi_rank
            rank = mpi_rank()
            import torch
            device_id = rank % torch.cuda.device_count()
            providers = [('CUDAExecutionProvider', {'device_id': device_id}), 'CPUExecutionProvider']
        except:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        # Load models
        self.det_sess = ort.InferenceSession(os.path.join(model_dir, 'det/inference.onnx'), providers=providers)
        self.rec_sess = ort.InferenceSession(os.path.join(model_dir, 'rec/inference.onnx'), providers=providers)
        self.docori_sess = ort.InferenceSession(os.path.join(model_dir, 'docori/inference.onnx'), providers=providers)
        self.lineori_sess = ort.InferenceSession(os.path.join(model_dir, 'textlineori/inference.onnx'), providers=providers)
        self.uvdoc_sess = ort.InferenceSession(os.path.join(model_dir, 'uvdoc/inference.onnx'), providers=providers)

    def load_keys(self, keys_path):
        with open(keys_path, 'r', encoding='utf-8') as f:
            keys = f.read().splitlines()
        keys = ['blank'] + keys + [' ']
        return keys

    def preprocess_det(self, img):
        # Resize to multiple of 32
        h, w = img.shape[:2]
        limit_side_len = 960
        ratio = 1.0
        if max(h, w) > limit_side_len:
            if h > w:
                ratio = float(limit_side_len) / h
            else:
                ratio = float(limit_side_len) / w
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)
        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)
        
        img_resize = cv2.resize(img, (resize_w, resize_h))
        img_resize = img_resize.astype('float32')
        
        # Revert to original PaddleOCR preprocessing (BGR, mean/std)
        # mean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375]
        img_resize = (img_resize - np.array([123.675, 116.28, 103.53], dtype=np.float32)) / np.array([58.395, 57.12, 57.375], dtype=np.float32)
        
        img_resize = img_resize.transpose((2, 0, 1))
        return img_resize[np.newaxis, :], ratio, w, h

    def postprocess_det(self, pred, ratio, src_w, src_h):
        pred = pred[0, 0, :, :]
        segmentation = pred > 0.3
        boxes, scores = self.boxes_from_bitmap(pred, segmentation, src_w, src_h)
        return boxes

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        bitmap = _bitmap
        height, width = bitmap.shape
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), 1000)
        boxes = []
        scores = []
        
        for i in range(num_contours):
            contour = contours[i]
            points, sside = self.get_mini_boxes(contour)
            if sside < 3:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if score < 0.5:
                continue
            
            box = self.unclip(points)
            box, sside = self.get_mini_boxes(box)
            if sside < 3 + 2:
                continue
            
            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int16))
            scores.append(score)
        return np.array(boxes), scores

    def unclip(self, box):
        unclip_ratio = 1.5
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded.reshape(-1, 1, 2)

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2
        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def preprocess_rec(self, img_crop):
        """
        动态调整识别图像尺寸。
        不再强制压缩到 320 像素，而是根据文本长短动态伸缩宽度。
        """
        h, w = img_crop.shape[:2]
        # PaddleOCR 默认输入高度为 48 (v3/v4/v5) 或 32 (v2)
        # 请根据您的模型确认，通常 v4/v5 是 48
        imgH = 48 
        
        # 计算保持比例后的新宽度
        ratio = w / float(h)
        resize_w = int(math.ceil(imgH * ratio))
        
        # 1. 动态宽度限制 (Dynamic Width)
        # 不要限制在 320！给长文本足够的空间。
        # 这里设置一个很大的上限（如 2000），防止极个别异常图片导致内存溢出
        if resize_w > 2000:
            resize_w = 2000
            
        # 2. 宽度对齐 (Padding/Alignment)
        # CRNN/SVTR 模型通常需要宽度是 32 的倍数（或者是卷积核步长的倍数）
        # 即使不是严格必须，对齐通常能提升推理速度
        resize_w = max(int(round(resize_w / 32) * 32), 32)
            
        # 3. 调整大小
        resized_image = cv2.resize(img_crop, (resize_w, imgH))
        resized_image = resized_image.astype('float32')
        
        # 4. 归一化 (Standard Rec Normalization)
        # 转换范围到 [-1, 1]
        resized_image = resized_image / 255.0
        resized_image = (resized_image - 0.5) / 0.5
        
        # 5. 通道转换 HWC -> CHW
        resized_image = resized_image.transpose((2, 0, 1))
        
        # 6. 添加 Batch 维度
        # 输出形状: [1, 3, 48, dynamic_width]
        return resized_image[np.newaxis, :]

    # def preprocess_rec(self, img_crop):
    #     # Based on predict_rec.py resize_norm_img
    #     # Default rec_image_shape is [3, 48, 320] for PP-OCRv3/v4/v5
    #     imgC, imgH, imgW = 3, 48, 320
    #     max_wh_ratio = imgW / imgH
        
    #     if math.ceil(imgH * ratio) > imgW:
    #         resized_w = imgW
    #     else:
    #         resized_w = int(math.ceil(imgH * ratio))
            
    #     resized_image = cv2.resize(img_crop, (resized_w, imgH))
    #     resized_image = resized_image.astype('float32')
    #     resized_image = resized_image.transpose((2, 0, 1)) / 255.0
    #     resized_image -= 0.5
    #     resized_image /= 0.5
        
    #     padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    #     padding_im[:, :, 0:resized_w] = resized_image
    #     return padding_im[np.newaxis, :]

    def decode_rec(self, preds):
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = ""
        for i in range(preds_idx.shape[1]):
            idx = preds_idx[0, i]
            if idx != 0 and idx < len(self.keys): # 0 is blank
                if i > 0 and idx == preds_idx[0, i-1]:
                    continue
                text += self.keys[idx]
        return text

    def preprocess_docori(self, img):
        img_resize = cv2.resize(img, (224, 224))
        img_resize = img_resize.astype('float32')
        img_resize = (img_resize - np.array([123.675, 116.28, 103.53], dtype=np.float32)) / np.array([58.395, 57.12, 57.375], dtype=np.float32)
        img_resize = img_resize.transpose((2, 0, 1))
        return img_resize[np.newaxis, :]

    def preprocess_lineori(self, img):
        img_resize = cv2.resize(img, (160, 80)) # 80x160
        img_resize = img_resize.astype('float32')
        img_resize = (img_resize - 127.5) / 127.5
        img_resize = img_resize.transpose((2, 0, 1))
        return img_resize[np.newaxis, :]

    def preprocess_uvdoc(self, img):
        # UVDoc usually expects 128x128 or similar, but let's check input shape
        # Assuming dynamic, but let's resize to 512x512 for now or keep original if supported
        # Based on polygraphy, it has dynamic input.
        # Let's resize to 512x512 for consistency
        img_resize = cv2.resize(img, (512, 512))
        img_resize = img_resize.astype('float32') / 255.0
        img_resize = img_resize.transpose((2, 0, 1))
        return img_resize[np.newaxis, :]

    def extract_text(self, image_input):
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                print(f"Failed to load image: {image_input}")
                return ""
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB for PaddleOCR models
        elif isinstance(image_input, np.ndarray):
            img = image_input
            # Assuming input is RGB if it comes from PIL -> np.array conversion in runner
            # But wait, cv2.imread reads BGR. My previous code converted BGR to RGB.
            # If I pass a numpy array, I need to know if it is RGB or BGR.
            # Let's assume the caller handles this or provides RGB.
            # If the caller provides PIL image converted to numpy, it is RGB.
        else:
            print("Unsupported image input type")
            return ""

        # print(f"Image loaded: {img.shape}")
        
        # 1. DocOri (Skipped to match predict_system.py baseline)
        # doc_input = self.preprocess_docori(img)
        # doc_out = self.docori_sess.run(None, {'x': doc_input})[0]
        # doc_idx = np.argmax(doc_out)
        # print(f"DocOri index: {doc_idx}")
        # if doc_idx == 1:
        #     img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # elif doc_idx == 2:
        #     img = cv2.rotate(img, cv2.ROTATE_180)
        # elif doc_idx == 3:
        #     img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 2. UVDoc (Skipped to match predict_system.py baseline)
        # try:
        #     uv_input = self.preprocess_uvdoc(img)
        #     uv_out = self.uvdoc_sess.run(None, {'image': uv_input})[0]
        #     uv_img = uv_out[0].transpose(1, 2, 0)
        #     if uv_img.max() <= 1.1:
        #         uv_img = uv_img * 255
        #     img = np.clip(uv_img, 0, 255).astype(np.uint8)
        # except Exception as e:
        #     print(f"UVDoc failed: {e}")

        # 3. Detection
        det_input, ratio, src_w, src_h = self.preprocess_det(img)
        try:
            det_out = self.det_sess.run(None, {'x': det_input})[0]
        except Exception as e:
            # print(f"Detection inference failed: {e}")
            return ""
            
        boxes = self.postprocess_det(det_out, ratio, src_w, src_h)
        # print(f"Detection found {len(boxes)} boxes")

        # 4. Recognition
        full_text = []
        if len(boxes) == 0:
            # print("No boxes found, trying whole image for recognition...")
            # Fallback: use whole image
            boxes = [np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]], dtype=np.float32)]

        boxes = self.sorted_boxes(boxes)
        
        for i, box in enumerate(boxes):
            # Crop
            pts = box.astype(np.float32)
            w = int(np.linalg.norm(pts[0] - pts[1]))
            h = int(np.linalg.norm(pts[0] - pts[3]))
            if w == 0 or h == 0: continue
            
            src_pts = pts
            dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            crop = cv2.warpPerspective(img, M, (w, h))

            # 5. LineOri (Optional, but kept as per previous request, though predict_system.py doesn't use it by default)
            # To strictly follow predict_system.py, we might skip this too, but let's keep it for robustness if it works.
            # For now, let's skip it to isolate issues.
            # line_input = self.preprocess_lineori(crop)
            # line_out = self.lineori_sess.run(None, {'x': line_input})[0]
            # if np.argmax(line_out) == 1:
            #     crop = cv2.rotate(crop, cv2.ROTATE_180)

            # Rec
            rec_input = self.preprocess_rec(crop)
            rec_out = self.rec_sess.run(None, {'x': rec_input})[0]
            # print(f"Rec out shape: {rec_out.shape}, Max prob: {np.max(rec_out)}")
            text = self.decode_rec(rec_out)
            # print(f"Rec text: '{text}'")
            full_text.append(text)

        # Normalize whitespace: replace newlines with spaces and collapse multiple spaces
        # Note: previous line returned newline-separated text; update to return single-line text
        result = " ".join([t.strip() for t in full_text if t.strip()])
        result = " ".join(result.split())
        return result

    def sorted_boxes(self, dt_boxes):
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                    _boxes[j + 1][0][0] < _boxes[j][0][0]
                ):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes

if __name__ == "__main__":
    ocr = OCRSystem('/home/xsf/fl/edge/models/ONNX', '/home/xsf/fl/edge/models/PaddleOCR/ppocr/utils/dict/ppocrv5_dict.txt')
    print(ocr.extract_text('/home/xsf/fl/edge/images/mtgp_en.png'))
