import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

def load_models():
    detector = YOLO("detection_model.pt")
    classifier = tf.keras.models.load_model("classification_model_beta.keras")
    class_names = ["Metal", "Organic", "Paper", "Plastic", "glass"]
    return detector, classifier, class_names

def process_image(image_rgb, detector, classifier, class_names):
    display_height, display_width = image_rgb.shape[:2]
    base_font_scale = min(display_width, display_height) / 1200
    font_scale = max(min(base_font_scale, 1.2), 0.6)
    text_thickness = max(2, min(3, int(font_scale * 2.5)))

    color_map = {
        'Metal': (255, 0, 0),      # Red
        'Organic': (0, 255, 0),    # Green
        'Paper': (0, 0, 255),      # Blue
        'Plastic': (255, 255, 0),  # Yellow
        'glass': (255, 0, 255)     # Magenta
    }

    results = detector.predict(image_rgb, conf=0.42, iou=0.45, imgsz=640)
    image_with_boxes = image_rgb.copy()
    class_stats = {cls: {'count': 0, 'confidence': 0} for cls in class_names}

    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            if x2 > x1 and y2 > y1:
                detected_object = image_rgb[y1:y2, x1:x2]
                if detected_object.size > 0:
                    obj_pil = Image.fromarray(detected_object)
                    img_resized = obj_pil.resize((224, 224))
                    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

                    predictions = classifier.predict(img_array, verbose=0)
                    predicted_class = class_names[np.argmax(predictions)]
                    confidence = float(np.max(predictions) * 100)

                    class_stats[predicted_class]['count'] += 1
                    class_stats[predicted_class]['confidence'] += confidence

                    color = color_map.get(predicted_class, (255, 255, 255))
                    cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{predicted_class} ({confidence:.0f}%)"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text_size = cv2.getTextSize(label, font, font_scale, text_thickness)[0]
                    
                    text_x = x1
                    text_y = max(y1 - 8, text_size[1] + 8)
                    
                    cv2.rectangle(image_with_boxes,
                                  (text_x - 4, text_y - text_size[1] - 6),
                                  (text_x + text_size[0] + 4, text_y + 6),
                                  (0, 0, 0), -1)
                    cv2.putText(image_with_boxes, label, (text_x, text_y),
                                font, font_scale, (255, 255, 255), text_thickness)
        total_objects = len(results[0].boxes)
    else:
        img_pil = Image.fromarray(image_rgb).resize((224, 224))
        img_array = np.expand_dims(np.array(img_pil) / 255.0, axis=0)

        predictions = classifier.predict(img_array, verbose=0)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = float(np.max(predictions) * 100)

        class_stats[predicted_class]['count'] = 1
        class_stats[predicted_class]['confidence'] = confidence

        label = f"{predicted_class} ({confidence:.0f}%)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, font_scale * 1.4, text_thickness + 1)[0]

        cv2.rectangle(image_with_boxes,
                      (8, 8),
                      (12 + text_size[0], 40 + text_size[1]),
                      (0, 0, 0), -1)
        cv2.putText(image_with_boxes, label, (10, 30 + text_size[1] // 2),
                    font, font_scale * 1.4, (255, 255, 255), text_thickness + 1)
        total_objects = 1

    return image_with_boxes, total_objects, class_stats
