import mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
from PIL import Image
import sys


photoPath = ""
outputPath = "default.png"

if len(sys.argv) == 2:
    photoPath = sys.argv[1]
elif len(sys.argv) == 3:
    photoPath = sys.argv[1]
    outputPath = sys.argv[2]

if "png" not in outputPath:
    outputPath+=".png"
elif "jpg" in outputPath:
    print("JPG is not supported")

modelPath = "models/selfie_multiclass_256x256.tflite"
baseOptions = python.BaseOptions(model_asset_path=modelPath)
runningMode = vision.RunningMode
imageSegmenter = vision.ImageSegmenter
imageSegmenterOptions = vision.ImageSegmenterOptions

options = imageSegmenterOptions(
    baseOptions,
    running_mode = runningMode.IMAGE,
    output_category_mask = True
)

with imageSegmenter.create_from_options(options) as segmenter:
    image = mediapipe.Image.create_from_file(photoPath)
    segmentation_result = segmenter.segment(image)
    category_mask = segmentation_result.category_mask

    image_data = image.numpy_view()
    bg_image = np.zeros(image_data.shape, dtype=np.uint8)
    bg_image[:] = (255,255,255,255)

    face_condition = np.stack((category_mask.numpy_view(),) * 4, axis=-1) == 3
    hair_condition = np.stack((category_mask.numpy_view(),) * 4, axis=-1) == 1
    full_condition = np.stack((category_mask.numpy_view(),) * 4, axis=-1) > 0

    segmented_image = np.where(full_condition, image_data, bg_image) # Image with background removed

    face_indices = np.argwhere(face_condition)
    y_mean_face, x_mean_face, n = face_indices.mean(axis=0) # Finding the centre of the face
    y_mean_face, x_mean_face = round(x_mean_face), round(y_mean_face)
    x_max_face, x_min_face = np.amax(face_indices[:, 1]), np.amin(face_indices[:, 1]) # For cropping
    y_max_face, y_min_face = np.amax(face_indices[:, 0]), np.amin(face_indices[:, 0])

    hair_indices = np.argwhere(hair_condition)
    if len(hair_indices) > 0:
        y_max_hair, y_min_hair = np.amax(hair_indices[:, 0]), np.amin(hair_indices[:, 0])
        x_max_hair, x_min_hair = np.amax(hair_indices[:, 1]), np.amin(hair_indices[:, 1])
    else:
        y_max_hair, y_min_hair = -1, -1
        x_max_hair, x_min_hair = -1, -1
    
    if len(hair_indices) > 0:
        total_height = round((y_max_face - y_min_hair)/0.75)
        total_width = round((x_max_hair - x_min_hair)/0.75)

        y_min = round(y_min_hair - total_height*0.05)
        y_max = round(y_max_face + total_height*0.2)

        x_min = round(x_min_hair - total_width*0.125)
        x_max = round(x_max_hair + total_width*0.125)
    else:
        total_height = round((y_max_face - y_min_face)/0.75)
        total_width = round((x_max_face - x_min_face)/0.75)

        y_min = round(y_min_face - total_height*0.05)
        y_max = round(y_max_face + total_height*0.2)

        x_min = round(x_min_face - total_width*0.125)
        x_max = round(x_max_face + total_width*0.125)

    y_max = round(y_max*((total_width/total_height)/0.77)) # Correct for aspect ratio

    cropped_image = segmented_image[y_min:y_max, x_min:x_max]

    img = Image.fromarray(cropped_image)
    img = img.resize((400,514), Image.Resampling.LANCZOS)
    img.save(outputPath)
