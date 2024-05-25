import os, cv2
import numpy as np
from pathlib import Path

class ThresholdbasedSegmentation:
    def fit(self, train_set: dict) -> None:
        images_file = os.listdir(train_set['images'])
        
        ratio_height = []
        ratio_size = []
        
        for image_file in images_file:
            label_file = os.path.join(train_set['labels'], Path(image_file).stem + '.txt')
            label = open(label_file, 'r')
            
            bboxes = [list(map(float, line.split())) for line in label.readlines()]
            for bbox in bboxes:
                ratio_height.append(bbox[4])
                ratio_size.append(bbox[3] / bbox[4])
        
        self.range_ratio_height = (np.min(ratio_height), np.max(ratio_height))
        self.range_ratio_size = (np.min(ratio_size), np.max(ratio_size))
        
    def predict(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, otsu_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(otsu_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        for contour in contours:
            x = contour[:, 0, 0]
            y = contour[:, 0, 1]
            
            (h, w) = image.shape[:2]
            min_x, max_x = x.min(), x.max()
            min_y, max_y = y.min(), y.max()
            box_w, box_h = max_x - min_x, max_y - min_y
            
            if box_h == 0 or box_w == 0:
                continue
            
            check_ratio_height = (self.range_ratio_height[0] <= box_h / h) and (box_h / h <= self.range_ratio_height[1])
            check_ratio_size = (self.range_ratio_size[0] <= box_w / box_h) and (box_w / box_h <= self.range_ratio_size[1])
            
            if check_ratio_height and check_ratio_size:
                bboxes.append([(min_x + max_x) / (2 * w), (min_y + max_y) / (2 * h), box_w / w, box_h / h])
                
        return bboxes
    
class EdgebasedSegmentation:
    def fit(self, train_set: dict) -> None:
        images_file = os.listdir(train_set['images'])
        
        ratio_height = []
        ratio_size = []
        
        for image_file in images_file:
            label_file = os.path.join(train_set['labels'], Path(image_file).stem + '.txt')
            label = open(label_file, 'r')
            
            bboxes = [list(map(float, line.split())) for line in label.readlines()]
            for bbox in bboxes:
                ratio_height.append(bbox[4])
                ratio_size.append(bbox[3] / bbox[4])
        
        self.range_ratio_height = (np.min(ratio_height), np.max(ratio_height))
        self.range_ratio_size = (np.min(ratio_size), np.max(ratio_size))
        
    def predict(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        brightness = 0
        contrast = 2
        enhanced_image = cv2.addWeighted(blur_image, contrast, np.zeros(blur_image.shape, blur_image.dtype), 0, brightness)
        
        edges = cv2.Canny(enhanced_image, 50, 100)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        for contour in contours:
            x = contour[:, 0, 0]
            y = contour[:, 0, 1]
            
            (h, w) = image.shape[:2]
            min_x, max_x = x.min(), x.max()
            min_y, max_y = y.min(), y.max()
            box_w, box_h = max_x - min_x, max_y - min_y
            
            if box_h == 0 or box_w == 0:
                continue
            
            check_ratio_height = (self.range_ratio_height[0] <= box_h / h) and (box_h / h <= self.range_ratio_height[1])
            check_ratio_size = (self.range_ratio_size[0] <= box_w / box_h) and (box_w / box_h <= self.range_ratio_size[1])
            
            if check_ratio_height and check_ratio_size:
                bboxes.append([(min_x + max_x) / (2 * w), (min_y + max_y) / (2 * h), box_w / w, box_h / h])
                
        return bboxes