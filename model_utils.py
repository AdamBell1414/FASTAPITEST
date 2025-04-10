import tensorflow.lite as tflite
import numpy as np
import json
import cv2
import os
from maize_leaf_detector import MaizeLeafClassifier

# Define constants
IMG_SIZE = 320
MODEL_PATH = "fall_armyworm_detector.tflite"

# Load class map
with open("class_map.json", "r") as f:
    CLASS_MAP = json.load(f)

# Reverse mapping from index to class name
IDX_TO_CLASS = {i: class_name for class_name, i in CLASS_MAP.items()}

class FallArmywormDetector:
    def __init__(self):
        # Initialize maize leaf classifier
        self.maize_classifier = MaizeLeafClassifier()
        
        # Load the TFLite model
        self.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        print("TFLite model size:", os.path.getsize(MODEL_PATH) / (1024 * 1024), "MB")

    def preprocess_image(self, image_path):
        """Preprocess the image for the model"""
        # Read image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB

        # Resize image to model input size
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        # Normalize pixel values
        image = image.astype(np.float32) / 255.0

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image

    def detect(self, image_path):
        """Run detection on an image"""
        # First, check if the image is a maize leaf
        maize_result = self.maize_classifier.classify(image_path)
        
        # If not a maize leaf, return early with a message
        if not maize_result["is_maize"]:
            return {
                "result": "Not a maize leaf",
                "description": "The uploaded image does not appear to be a maize leaf. Please upload an image of a maize plant.",
                "confidence": round(maize_result["confidence"] * 100, 2),
                "is_maize": False
            }
        
        # If it is a maize leaf, continue with fall armyworm detection
        image = self.preprocess_image(image_path)

        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], image)

        # Run inference
        self.interpreter.invoke()

        # Get output tensors - print shapes for debugging
        print("Output tensor shapes:")
        for i, output in enumerate(self.output_details):
            tensor = self.interpreter.get_tensor(output['index'])
            print(f"Output {i}: shape={tensor.shape}")

        # Get output tensors
        # The exact indices might need adjustment based on your model's output format
        try:
            # Try different output combinations
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])

            # Print shapes for debugging
            print(f"Boxes shape: {boxes.shape}")
            print(f"Classes shape: {classes.shape}")
            print(f"Scores shape: {scores.shape}")

            # Process detections
            detections = self.process_detections(boxes, classes, scores)

            # Determine final classification
            final_classification = self.determine_final_class(detections)

            # Create user-friendly result
            result = self.create_user_friendly_result(final_classification)
            
            # Add maize classification info
            result["is_maize"] = True
            result["maize_confidence"] = round(maize_result["confidence"] * 100, 2)

            return result

        except Exception as e:
            print(f"Error with default output order: {e}")

        # Try alternative output order
        try:
            boxes = self.interpreter.get_tensor(self.output_details[1]['index'])
            classes = self.interpreter.get_tensor(self.output_details[0]['index'])
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])

            # Print shapes for debugging
            print(f"Alternative - Boxes shape: {boxes.shape}")
            print(f"Alternative - Classes shape: {classes.shape}")
            print(f"Alternative - Scores shape: {scores.shape}")

            # Process detections
            detections = self.process_detections(boxes, classes, scores)

            # Determine final classification
            final_classification = self.determine_final_class(detections)

            # Create user-friendly result
            result = self.create_user_friendly_result(final_classification)
            
            # Add maize classification info
            result["is_maize"] = True
            result["maize_confidence"] = round(maize_result["confidence"] * 100, 2)

            return result

        except Exception as e2:
            print(f"Error with alternative output order: {e2}")
            raise e2

    def create_user_friendly_result(self, classification):
        """Create a user-friendly result message"""
        class_name = classification["class"]
        confidence = classification["confidence"]
        confidence_percent = round(confidence * 100, 2)

        result = {
            "confidence": confidence_percent
        }

        if class_name == "fall-armyworm-larval-damage":
            result["result"] = "Fall Armyworm larval damage detected"
            result["description"] = "Your maize plant shows signs of Fall Armyworm larval feeding damage"
        elif class_name == "fall-armyworm-egg":
            result["result"] = "Fall Armyworm eggs detected"
            result["description"] = "Fall Armyworm eggs have been detected on your maize plant"
        elif class_name == "fall-armyworm-frass":
            result["result"] = "Fall Armyworm frass detected"
            result["description"] = "Fall Armyworm frass (excrement) has been detected on your maize plant"
        elif class_name == "healthy-maize":
            result["result"] = "Healthy maize plant"
            result["description"] = "Your maize plant appears to be healthy with no signs of Fall Armyworm infestation"
        else:
            result["result"] = "Unknown condition"
            result["description"] = "Unable to determine the condition of your maize plant"

        return result

    def process_detections(self, boxes, classes, scores, threshold=0.5):
        """Process the raw detection results"""
        results = []

        # Ensure we're working with the right dimensions
        # If boxes has shape (1, N, 4), take the first batch
        if len(boxes.shape) == 3:
            boxes = boxes[0]

        # If classes has shape (1, N), take the first batch
        if len(classes.shape) == 2:
            classes = classes[0]
        elif len(classes.shape) == 3: # For one-hot encoded classes (1, N, num_classes)
            classes = classes[0]

        # If scores has shape (1, N), take the first batch
        if len(scores.shape) == 2:
            scores = scores[0]
        elif len(scores.shape) == 3: # For scores with extra dimension (1, N, 1)
            scores = scores[0, :, 0] if scores.shape[2] == 1 else scores[0]

        print(f"After reshaping - Boxes: {boxes.shape}, Classes: {classes.shape}, Scores: {scores.shape}")

        # Get the number of detections
        num_detections = min(len(boxes), len(scores))

        print(f"Processing {num_detections} detections")

        # Process each detection
        for i in range(num_detections):
            try:
                # Get score - ensure it's a scalar
                score = float(scores[i]) if np.isscalar(scores[i]) else float(scores[i].item())

                # Skip low confidence detections
                if score < threshold:
                    continue

                # Get class - handle different class formats
                if len(classes.shape) == 1: # Classes is a 1D array of indices
                    class_idx = int(classes[i]) if np.isscalar(classes[i]) else int(classes[i].item())
                else: # Classes is a 2D array, possibly one-hot encoded
                    class_idx = int(np.argmax(classes[i]))

                class_name = IDX_TO_CLASS.get(class_idx, f"Unknown-{class_idx}")

                # Get box coordinates - ensure they're scalars
                box = boxes[i]
                box_list = [float(coord) if np.isscalar(coord) else float(coord.item()) for coord in box]

                # Add detection to results
                results.append({
                    "class": class_name,
                    "confidence": score,
                    "box": box_list
                })
            except Exception as e:
                print(f"Error processing detection {i}: {e}")
                continue

        return results

    def determine_final_class(self, detections):
        """Determine the final class for the image based on detections"""
        if not detections:
            return {
                "class": "unknown",
                "confidence": 0.0
            }

        # Group detections by class
        detections_by_class = {}
        for detection in detections:
            class_name = detection["class"]
            if class_name not in detections_by_class:
                detections_by_class[class_name] = []
            detections_by_class[class_name].append(detection)

        # Define priority classes (fall armyworm stages)
        priority_classes = ["fall-armyworm-larval-damage", "fall-armyworm-egg"]

        # Check if any priority classes are detected
        for priority_class in priority_classes:
            if priority_class in detections_by_class:
                # Get the highest confidence detection for this class
                best_detection = max(detections_by_class[priority_class], key=lambda x: x["confidence"])
                return {
                    "class": priority_class,
                    "confidence": best_detection["confidence"]
                }

        # If no priority classes, check for healthy-maize
        if "healthy-maize" in detections_by_class:
            best_detection = max(detections_by_class["healthy-maize"], key=lambda x: x["confidence"])
            return {
                "class": "healthy-maize",
                "confidence": best_detection["confidence"]
            }

        # If only frass or nothing else recognized, return frass or unknown
        if "fall-armyworm-frass" in detections_by_class:
            best_detection = max(detections_by_class["fall-armyworm-frass"], key=lambda x: x["confidence"])
            return {
                "class": "fall-armyworm-frass",
                "confidence": best_detection["confidence"]
            }

        # Fallback to highest confidence detection of any class
        highest_conf_detection = max(detections, key=lambda x: x["confidence"])
        return {
            "class": highest_conf_detection["class"],
            "confidence": highest_conf_detection["confidence"]
        }

# Initialize the detector
detector = FallArmywormDetector()
