from src import detect_faces, show_bboxes
from PIL import Image
img = Image.open('images/office1.jpg')
bounding_boxes, landmarks = detect_faces(img)
show_bboxes(img, bounding_boxes, landmarks)
