from pytesseract import pytesseract
import cv2
import numpy as np
import tkinter as tk
from ultralytics import YOLO
import matplotlib.pyplot as plt

class OCR:
    def __init__(self):
        self.path = r'/opt/homebrew/bin/tesseract'
        pytesseract.tesseract_cmd = self.path

    def extract(self, filename):
        try:
            # Load the image using OpenCV
            image = cv2.imread(filename)
            if image is None:
                raise FileNotFoundError(f"Image at {filename} not found.")

            # Perform OCR on the image
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

            # Extract and print the bounding box coordinates for each word
            n_boxes = len(data['level'])
            result = []
            for i in range(n_boxes):
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                word = data['text'][i]
                if word.strip():  # Ensure we don't process empty strings
                    top_left = (x, y)
                    bottom_right = (x + w, y + h)
                    bounding_box = {'word': word, 'top_left': top_left, 'bottom_right': bottom_right}
                    result.append(bounding_box)

            return result
        except Exception as e:
            print(f"Error: {e}")
            return "Error"

def display_image_with_bounding_boxes(image_path, face_coords, text_coords):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image at {image_path} not found.")
        return

    # Draw face bounding boxes
    for (top_left, bottom_right) in face_coords:
        cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)

    # Draw text bounding boxes and labels
    for bounding_box in text_coords:
        word = bounding_box['word']
        top_left = bounding_box['top_left']
        bottom_right = bounding_box['bottom_right']

        # Ensure coordinates are within the image bounds
        h, w = image.shape[:2]
        if 0 <= top_left[0] < w and 0 <= top_left[1] < h and 0 <= bottom_right[0] < w and 0 <= bottom_right[1] < h:
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(image, word, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        else:
            print(f"Skipping word '{word}' with coordinates {top_left} to {bottom_right} - out of bounds.")

    # Get screen size using tkinter
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    # Calculate the aspect ratio of the image
    aspect_ratio = w / h

    # Resize image to fit within screen dimensions while maintaining aspect ratio
    if screen_width / screen_height < aspect_ratio:
        new_width = screen_width
        new_height = int(screen_width / aspect_ratio)
    else:
        new_height = screen_height
        new_width = int(screen_height * aspect_ratio)

    resized_image = cv2.resize(image, (new_width, new_height))

    # Create a window with the correct size
    cv2.namedWindow('Image with Bounding Boxes', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Image with Bounding Boxes', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Create a black canvas for the full screen
    canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    # Center the resized image on the canvas
    x_offset = (screen_width - new_width) // 2
    y_offset = (screen_height - new_height) // 2
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image

    # Display the image
    cv2.imshow('Image with Bounding Boxes', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_faces(image_path):
    # Load the YOLO v8 model
    model = YOLO('yolov8n.pt')  # Replace with your model path if custom

    # Load and preprocess the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Run the model on the image
    results = model(img_rgb)

    face_coords = []
    # Extract information from results
    for result in results:
        boxes = result.boxes.xyxy  # Bounding boxes

        # Annotate image
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            top_left = (x1, y1)
            bottom_right = (x2, y2)
            face_coords.append((top_left, bottom_right))
            print(f"Face found at Top-left: {top_left}, Bottom-right: {bottom_right}\n")

    return face_coords

if __name__ == "__main__":
    # Define the path to the image
    image_path = r'Aaadhaar-Update-process.jpg'

    # Face detection
    face_coords = detect_faces(image_path)

    # OCR
    ocr = OCR()
    text_coords = ocr.extract(image_path)

    # Perform OCR on the image and print text content
    text = pytesseract.image_to_string(image_path)
    print("All text detected in the picture:")
    print(text)

    if text_coords != "Error":
        print("Coordinates for every single word detected:")
        for bounding_box in text_coords:
            word = bounding_box['word']
            top_left = bounding_box['top_left']
            bottom_right = bounding_box['bottom_right']
            print(f"word: {word}, top_left: {top_left}, bottom_right: {bottom_right}")

        # Display the combined image with both face and text bounding boxes
        display_image_with_bounding_boxes(image_path, face_coords, text_coords)
