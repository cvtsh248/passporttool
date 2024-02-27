# A mediapipe powered passport photo generator 
Uses mediapipe's multiclass selfie segmentation model to produce a passport sized photo as per Singapore passport requirements.

## Usage
Run main.py as such:
``` python3 main.py path/to/image path/to/output ```

Note that the output argument is optional, and the program always outputs a .png file.

When taking your photo, ensure you are looking straight at the camera. Ensure that your entire head is visible, with a buffer region around it.

## Dependencies
* mediapipe
* numpy
* PIL
* sys
