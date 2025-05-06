import os
import sys
import tempfile
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import trimesh
import io
import base64
from pathlib import Path

# Import our Photo3DGenerator class
# Ensuring the script can be run from different directories
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import the Photo3DGenerator from our existing file
# Assuming the Photo3DGenerator class is defined in a file named photo_to_3d.py
from imgto  import Photo3DGenerator

# Create temporary directories for input and output
temp_input_dir = tempfile.mkdtemp()
temp_output_dir = tempfile.mkdtemp()

# Initialize our generator with the temp directories
generator = Photo3DGenerator(input_dir=temp_input_dir, output_dir=temp_output_dir)

# Set up Streamlit app
st.set_page_config(page_title="Photo to 3D Model Converter", layout="wide")

# Add a title and description
st.title("Photo to 3D Model Converter")
st.markdown("""
This app converts a 2D photo of a single object into a 3D model. 
Upload an image, and the app will generate a 3D model that you can download and view.
""")

# Create columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Save the uploaded image to the temp input directory
        img_path = os.path.join(temp_input_dir, uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Create a button to start processing
        if st.button("Generate 3D Model"):
            with st.spinner("Processing image... This may take a minute."):
                try:
                    # Process the image using our generator
                    results = generator.process_image(img_path)
                    
                    # Store the results for display
                    st.session_state.results = results
                    st.session_state.processed = True
                    
                    # Success message
                    st.success("3D model generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error processing image: {e}")

# Display results in the second column
with col2:
    if 'processed' in st.session_state and st.session_state.processed:
        # Get results from session state
        results = st.session_state.results
        
        # Create tabs for different visualizations
        tabs = st.tabs(["3D Model", "Depth Map", "Results"])
        
        # Tab 1: 3D Model Visualization using Plotly
        with tabs[0]:
            # Load the mesh
            mesh = trimesh.load(results['mesh_obj'])
            vertices = mesh.vertices
            faces = mesh.faces
            
            # Create a plotly figure
            fig = go.Figure(data=[
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    opacity=0.8,
                    colorscale='Viridis',
                )
            ])
            
            # Update layout for better visualization
            fig.update_layout(
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            )
            
            # Display the 3D model
            st.plotly_chart(fig, use_container_width=True)
        
        # Tab 2: Depth Map
        with tabs[1]:
            # Load and display the depth map
            depth_map = plt.imread(results['depth_map'])
            st.image(depth_map, caption="Depth Map", use_column_width=True)
        
        # Tab 3: Download links
        with tabs[2]:
            st.subheader("Download Files")
            
            # Function to create a download link
            def get_binary_file_downloader_html(bin_file, file_label='File'):
                with open(bin_file, 'rb') as f:
                    data = f.read()
                bin_str = base64.b64encode(data).decode()
                href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
                return href
            
            # Create download links for STL and OBJ files
            st.markdown(get_binary_file_downloader_html(results['mesh_stl'], 'STL File'), unsafe_allow_html=True)
            st.markdown(get_binary_file_downloader_html(results['mesh_obj'], 'OBJ File'), unsafe_allow_html=True)
            
            # Create download link for visualization
            st.markdown(get_binary_file_downloader_html(results['visualization'], 'Visualization Image'), unsafe_allow_html=True)
            
            # Additional information
            st.info("""
            - STL files can be used for 3D printing
            - OBJ files can be loaded into most 3D modeling software
            - You can view the 3D model online using services like [3D Viewer Online](https://www.3dvieweronline.com/)
            """)

# Add information about the app
st.markdown("---")
st.subheader("How It Works")
st.markdown("""
1. The app processes your image and removes the background
2. A depth estimation AI model generates a depth map
3. The depth map is converted to a point cloud
4. A 3D mesh is created from the point cloud
5. The mesh is saved in STL and OBJ formats

**Note:** This is a prototype. Results will vary depending on the quality and clarity of the input image.
""")

# Add some helpful tips
st.subheader("Tips for Best Results")
st.markdown("""
- Use images with a single, clearly defined object
- Ensure the object has good lighting and contrast with the background
- Simple objects with clear edges work best
- Avoid very complex or transparent objects
""")