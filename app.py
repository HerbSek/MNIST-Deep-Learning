import torch
from cnn_model import cnn
import streamlit as st
import cv2
import numpy as np

def load_model():
    model = cnn()
    model.load_state_dict(torch.load('./model.pth'))
    model.eval()
    return model

model = load_model()


image = cv2.imread("mnist_test/test/009991-num8.png")

with st.sidebar:
    st.write("hello")
    file = st.file_uploader(" ", type = ["jpg", "webp"])
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    

if image is None:
    st.write("Error: Image not loaded. Check the file path.")
else:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_rgb = cv2.resize(image_rgb, (28,28), interpolation=cv2.INTER_AREA)
    st.image(image_rgb, caption='Loaded Image', use_column_width=True)
    image_rgb_array = np.array(image_rgb)
    tensor_img = torch.tensor(image_rgb_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    st.write(tensor_img.shape)
    output = model(tensor_img)
    prediction = torch.argmax(output , dim=1)
    st.write(prediction.item())
    print("done")





