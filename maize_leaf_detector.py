import tensorflow.lite as tflite
import numpy as np
import cv2

class MaizeLeafClassifier:
    def __init__(self, model_path="maizeleafclassifier2_metadata.tflite"):
        # Load the TFLite model
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Define class mapping
        self.class_map = {0: "Maize", 1: "Not_Maize"}
        
        # Get input shape
        self.input_shape = self.input_details[0]['shape'][1:3]  # Height, width
        print(f"Maize classifier input shape: {self.input_shape}")

    def preprocess_image(self, image_path):
        """Preprocess the image for the maize leaf classifier model"""
        # Read image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Resize image to model input size
        image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))

        # Normalize pixel values
        image = image.astype(np.float32) / 255.0

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image

    def classify(self, image_path):
        """Classify if the image contains a maize leaf"""
        image = self.preprocess_image(image_path)

        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], image)

        # Run inference
        self.interpreter.invoke()

        # Get output tensor
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(output[0])
        confidence = output[0][predicted_class_idx]
        
        result = {
            "is_maize": predicted_class_idx == 0,  # True if class 0 (Maize)
            "class": self.class_map[predicted_class_idx],
            "confidence": float(confidence)
        }
        
        return result
