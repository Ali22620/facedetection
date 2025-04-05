import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
#create application title and file uploader widget
st.title("OpenCV Deep learning Based face Detection")
img_file_buffer=st.file_uploader("Chose a file", type=['jpg','jpeg','png'])

#function for detecting faces in an image
def detectFaceOpenCVDnn(net,frame):
    # create a blob from the image and apply same pre-processing
    blob=cv2.dnn.blobFromImage(frame, 1.0, (300,300),[104,117,123],False,False)
    # set the blob as input model
    net.setInput(blob)
    # get detections
    detections=net.forward()
    return detections


#Function for anotating the image with bounding boxes for each detected face.
def process_detections(frame, detections, conf_threshold=0.5):
    bboxes=[]
    frame_h=frame.shape[0]
    frame_w=frame.shape[1]
    # loop all detections and draw bounding boxes around each face
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        x1 = int(detections[0, 0, i, 3] * frame_w)
        y1 = int(detections[0, 0, i, 4] * frame_h)
        x2 = int(detections[0, 0, i, 5] * frame_w)
        y2 = int(detections[0, 0, i, 6] * frame_h)
        bboxes.append([x1,y1,x2,y2])
        bb_line_thickness = max(1, int(round(frame_h / 2000)))
        # Draw bounding boxes around detected faces.
        cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), bb_line_thickness, cv2.LINE_8)
    return frame,bboxes

#Function to load the DNN Model
@st.cache_resource
def load_model():
    configFile = "deploy.prototxt"
    modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net


# function to generate a download link for output file
def get_image_download_link(img, filename, text):
    buffered=BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download = "{filename}">{text}</a>'
    return href

net=load_model()

if img_file_buffer is not None:
    # read file and convert it in to open cv image
    raw_bytes=np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    # loads image in a BGR Channel
    image=cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    # or use PIL image (Which uses an RGB order)
    # image=np.array(Image.open(img_file_buffer))

    # Create Place holders to display input and output Image
    placeholders=st.columns(2)
    # Display input image in first place holder
    placeholders[0].image(image, channels='BGR')
    placeholders[0].text("Input Image")

    # Create a slider and get the threshold from slider
    conf_threshold= st.slider("Set Confidence Threshold", min_value=0.0, max_value=1.0, step=0.01, value=0.5)

    # Call the Face detection Model to detect faces in the image
    detections=detectFaceOpenCVDnn(net,image)

    # Process the detections based on the current confidence threshold
    out_image, _= process_detections(image, detections, conf_threshold=conf_threshold)

    # Display the Detected faces
    placeholders[1].image(out_image, channels='BGR')
    placeholders[1].text("Output Image")

    # Convert opencv image to PIL
    out_image=Image.fromarray(out_image[:,:,::-1])

    # Create a link for downloading output file.
    st.markdown(get_image_download_link(out_image, "Face_output.jpg", 'Download Output Image'), unsafe_allow_html=True)

