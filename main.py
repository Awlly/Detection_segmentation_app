import streamlit as st
import numpy as np
import torch
import torch.nn as nn

from torchvision import transforms as T
from PIL import Image
from PIL import ImageDraw, ImageFont
from io import BytesIO
from denoiser_MODEL import UNet, Down, Up, ConvBlock
from semantic_seg_MODEL import SemUNet

def main():
    menu = ["Detection of brain tumors", "Image Denoiser", "Semantic forest segmentation"]
    choice = st.sidebar.radio("Навигация", menu)

    if choice == "Detection of brain tumors":
        page_tumor_detection()
    elif choice == "Image Denoiser":
        page_document_cleanup()
    elif choice == "Semantic forest segmentation":
        page_semantic_segmentation()



def page_tumor_detection():
    @st.cache_data

    def load_model():
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', source='github')
        return model
    model = load_model()

    st.title("YOLOv5 detection of brain tumors")

    uploaded_file = st.file_uploader("Upload an image and I will try to determine the type of tumor", type=["jpg", "png", "jpeg"])

    def draw_boxes_pil(image, results):
        draw = ImageDraw.Draw(image)
        labels = results.xyxyn[0][:, -1].cpu().numpy()
        cord = results.xyxyn[0][:, :-1].cpu().numpy()
        for i in range(len(labels)):
            row = cord[i]
            x1, y1, x2, y2 = int(row[0]*image.width), int(row[1]*image.height), int(row[2]*image.width), int(row[3]*image.height)
            box_color = (0, 255, 0)  # green box
            try:
                font = ImageFont.truetype("arial.ttf", 40)
            except IOError:
                font = ImageFont.load_default()
            label = f'{model.names[int(labels[i])]} {row[4]:.2f}'
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
            draw.text((x1, y1), label, fill=box_color, font=font)
            return image

    if uploaded_file is not None:
        image = Image.open(BytesIO(uploaded_file.read())).convert('RGB')
        results = model(image)
        image_boxed = draw_boxes_pil(image, results)
        st.image(image_boxed, caption='Processed Image', use_column_width=True)


def page_document_cleanup():
    model = UNet()

    @st.cache_data
    def load_model(model_path):
        model = UNet()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model

    def transform_image(image):
        transform = T.Compose([
        T.Resize((256, 256)),  
        T.Grayscale(num_output_channels=1),  
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
        ])
        return transform(image).unsqueeze(0)

    def show_output_image(output_tensor):
        output_image = output_tensor.squeeze().detach().numpy()
        output_image = (output_image * 255).astype(np.uint8)
        output_image = Image.fromarray(output_image)
        return output_image

    st.title('Image Denoiser')
    model_path = 'denoising_unet.pth'
    model = load_model(model_path)

    uploaded_file = st.file_uploader("Upload an image and I'll try to remove noise from it", type="png")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.write("Denoising in progress...")
        input_tensor = transform_image(image)
        with torch.no_grad():
            output_tensor = model(input_tensor)
        denoised_image = show_output_image(output_tensor)
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)

        with col2:
            st.image(denoised_image, caption='Denoised Image', use_column_width=True)

            
    
def page_semantic_segmentation():

    in_channels = 3  
    out_channels = 1  
    model = SemUNet(in_channels, out_channels)
    model.load_state_dict(torch.load('semseg_best.pth', map_location=torch.device('cpu')))
    model.eval()
    
    st.title('Semantic forest segmentation with U-Net (4 layers)')
    threshold = st.slider('Select a threshold value', 
                        min_value=0.30, 
                        max_value=0.60, 
                        value=0.45,  
                        step=0.00001, 
                        format='%f')

    def transform_image(image):
        transform = T.Compose([
            T.Resize((256, 256)),   
            T.ToTensor(),
        ])
        return transform(image).unsqueeze(0)

    def process_image(image_path, threshold):
        image = Image.open(image_path).convert("RGB")
        original_image_np = np.array(image)
        image_tensor = transform_image(image)

        with torch.no_grad():
            prediction = model(image_tensor)
        predicted_mask = torch.sigmoid(prediction).data.numpy()
        predicted_mask = (predicted_mask > threshold).astype(np.uint8)
        predicted_mask = np.squeeze(predicted_mask, axis=(0, 1))

        predicted_mask = Image.fromarray(predicted_mask)
        predicted_mask = predicted_mask.resize((original_image_np.shape[1], original_image_np.shape[0]), resample=Image.NEAREST)
        predicted_mask = np.array(predicted_mask)

        violet_mask = np.zeros_like(original_image_np)
        violet_mask[:, :, 0] = predicted_mask * 238
        violet_mask[:, :, 2] = predicted_mask * 130
        overlay_image_np = np.where(predicted_mask[..., None], violet_mask, original_image_np).astype(np.uint8)

        return original_image_np, overlay_image_np


    uploaded_file = st.file_uploader("Upload an image...", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        st.session_state['uploaded_file'] = uploaded_file

    if 'uploaded_file' in st.session_state:
        original_image, overlay_image = process_image(st.session_state['uploaded_file'], threshold)
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, use_column_width=True)
            st.caption('Original Image')
        with col2:
            st.image(overlay_image, use_column_width=True)
            st.caption('Segmented Image')

if __name__ == "__main__":
    main()





    











