import streamlit as st
import requests
import base64
import cv2
import numpy as np
import io
from PIL import Image

st.title("Image Detection with Roboflow API using OpenCV")
st.write("Upload an image and see the response from Roboflow API.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()

    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR) 

    # st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption='Uploaded Image', use_column_width=True)

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    url = "https://detect.roboflow.com/practicum-t7bs6/1?api_key=X1OU6EeTNHOU8Amt05Hi"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }

    response = requests.post(url, data=image_base64, headers=headers)

    if response.status_code == 200:
        st.write("Detection Result:")
        response_data = response.json()

        for prediction in response_data['predictions']:
            x = prediction['x']
            y = prediction['y']
            width = prediction['width']
            height = prediction['height']
            class_name = prediction['class']
            confidence = prediction['confidence']

            top_left = (int(x - width / 2), int(y - height / 2))
            bottom_right = (int(x + width / 2), int(y + height / 2))

            cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 3)

            label = f"{class_name} ({confidence:.2f})"
            cv2.putText(img, label, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption='Detected Objects', use_column_width=True)

    else:
        st.write("Error:", response.status_code)
