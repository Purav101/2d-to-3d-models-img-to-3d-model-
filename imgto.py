import os
import numpy as np
import cv2
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from PIL import Image

# For background removal
import torch.nn.functional as F
from torchvision import transforms

# For depth estimation
from transformers import DPTForDepthEstimation, DPTFeatureExtractor

class Photo3DGenerator:
    def __init__(self, input_dir='inputs', output_dir='outputs'):
        """Initialize the Photo to 3D Model generator"""
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create directories if they don't exist
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize models
        self._init_models()

    def _init_models(self):
        """Initialize required Aopen sourse modls """
        print("Loading models...")
        
        # Load background removal model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize depth estimation model from Hugging Face
        self.depth_feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
        self.depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
        self.depth_model.to(self.device)
        
        print("Models loaded successfully")

    def process_image(self, image_path):
        """Process an input image and generate a 3D model"""
        print(f"Processing image: {image_path}")
        
        # Load and preprocess the image
        image = self._load_image(image_path)
        
        # Remove background
        masked_image = self._remove_background(image)
        
        # Generate depth map
        depth_map = self._generate_depth_map(masked_image)
        
        # Create point cloud from depth map
        point_cloud = self._create_point_cloud(depth_map, masked_image)
        
        # Generate mesh
        mesh = self._generate_mesh(point_cloud)
        
        # Save outputs
        filename = Path(image_path).stem
        self._save_outputs(mesh, filename, depth_map, masked_image)
        
        return {
            'mesh_stl': os.path.join(self.output_dir, f"{filename}.stl"),
            'mesh_obj': os.path.join(self.output_dir, f"{filename}.obj"),
            'depth_map': os.path.join(self.output_dir, f"{filename}_depth.png"),
            'visualization': os.path.join(self.output_dir, f"{filename}_visualization.png")
        }

    def _load_image(self, image_path):
        """Load and preprocess the image"""
        # Check if the file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image for consistency
        image = cv2.resize(image, (512, 512))
        
        return image

    def _remove_background(self, image):
        """Remove background from image using segmentation"""
        # This is a simplified version. In a full implementation,
        # you would use a proper segmentation model like U2Net or Segment Anything Model

        # For demonstration, we'll use a simple color-based segmentation
        # In a real application, replace this with a deep learning model
        
        # Convert to PIL Image for transforms
        pil_image = Image.fromarray(image)
        
        # Apply transformations for the model
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        input_tensor = preprocess(pil_image)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        
        # In a real implementation, you would run inference with a segmentation model here
        # For this prototype, we'll simulate a mask
        
        # Simplified approach: Use canny edge detection + dilate/erode to create a rough mask
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # Dilate to connect edges
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=3)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask from the largest contour
        mask = np.zeros_like(gray)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        
        # Apply mask to the original image
        mask_3channel = np.stack([mask, mask, mask], axis=2) / 255.0
        masked_image = (image * mask_3channel).astype(np.uint8)
        
        return masked_image

    def _generate_depth_map(self, image):
        """Generate depth map from image using DPT model"""
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)
        
        # Prepare image for the model
        inputs = self.depth_feature_extractor(images=pil_image, return_tensors="pt").to(self.device)
        
        # Get depth prediction
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to get the original image size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        # Convert to numpy
        depth_map = prediction.cpu().numpy()
        
        # Normalize depth values
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
        
        return normalized_depth

    def _create_point_cloud(self, depth_map, image):
        """Create a point cloud from depth map and color image"""
        h, w = depth_map.shape
        
        # Create coordinate grid
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Create mask for valid points (non-zero in masked image)
        mask = np.sum(image, axis=2) > 0
        
        # Filter points
        points_x = x[mask]
        points_y = y[mask] 
        points_z = depth_map[mask]
        
        # Normalize coordinates to [-1, 1]
        points_x = (points_x / w) * 2 - 1
        points_y = (points_y / h) * 2 - 1
        
        # Create point cloud
        points = np.column_stack([points_x, points_y, points_z])
        
        # Get colors
        colors = image[mask] / 255.0
        
        return {
            'points': points,
            'colors': colors
        }

    def _generate_mesh(self, point_cloud):
        """Generate a 3D mesh from point cloud data"""
        # This is a simplified approach using Ball-Pivoting algorithm
        # In a complete implementation, consider using more advanced 
        # algorithms like Poisson surface reconstruction
        
        # Create a point cloud object
        points = point_cloud['points']
        colors = point_cloud['colors']
        
        # Simple approach: Create a height field mesh (2.5D representation)
        # Extract x, y coordinates and z values
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        
        # Create a grid for the height field
        grid_size = 64
        grid_x = np.linspace(min(x), max(x), grid_size)
        grid_y = np.linspace(min(y), max(y), grid_size)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        
        # Create vertices
        vertices = []
        for i in range(grid_size):
            for j in range(grid_size):
                vertices.append([grid_x[i, j], grid_y[i, j], 0])  # Will update z later
        
        vertices = np.array(vertices)
        
        # Create faces (triangulation of the grid)
        faces = []
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                idx = i * grid_size + j
                faces.append([idx, idx + 1, idx + grid_size])
                faces.append([idx + 1, idx + grid_size + 1, idx + grid_size])
        
        faces = np.array(faces)
        
        # Create a mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        return mesh

    def _save_outputs(self, mesh, filename, depth_map, masked_image):
        """Save the generated outputs"""
        # Save the mesh as STL
        stl_path = os.path.join(self.output_dir, f"{filename}.stl")
        mesh.export(stl_path)
        
        # Save the mesh as OBJ
        obj_path = os.path.join(self.output_dir, f"{filename}.obj")
        mesh.export(obj_path)
        
        # Save the depth map
        depth_map_path = os.path.join(self.output_dir, f"{filename}_depth.png")
        plt.imsave(depth_map_path, depth_map, cmap='plasma')
        
        # Save the visualization
        vis_path = os.path.join(self.output_dir, f"{filename}_visualization.png")
        self._save_visualization(mesh, depth_map, masked_image, vis_path)
        
        print(f"Saved outputs to {self.output_dir}")

    def _save_visualization(self, mesh, depth_map, masked_image, output_path):
        """Create and save a visualization of the 3D model"""
        fig = plt.figure(figsize=(15, 5))
        
        # Original image
        ax1 = fig.add_subplot(131)
        ax1.imshow(masked_image)
        ax1.set_title("Masked Image")
        ax1.axis('off')
        
        # Depth map
        ax2 = fig.add_subplot(132)
        ax2.imshow(depth_map, cmap='plasma')
        ax2.set_title("Depth Map")
        ax2.axis('off')
        
        # 3D visualization
        ax3 = fig.add_subplot(133, projection='3d')
        # Plot the mesh
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Create a triangular mesh plot
        x = vertices[:, 0]
        y = vertices[:, 1]
        z = vertices[:, 2]
        
        # Plot the triangles
        tri = Axes3D.plot_trisurf(ax3, x, y, z, triangles=faces, cmap='viridis', alpha=0.7)
        
        ax3.set_title("3D Model")
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


def main():
    """Main function to demonstrate usage"""
    # Initialize the generator
    generator = Photo3DGenerator()
    
    # Example usage
    print("Please place your image in the 'inputs' directory")
    image_name = input("Enter the filename of your image (e.g., chair.jpg): ")
    image_path = os.path.join('inputs', image_name)
    
    try:
        # Process the image
        results = generator.process_image(image_path)
        
        print("\nProcessing complete!")
        print(f"STL file saved to: {results['mesh_stl']}")
        print(f"OBJ file saved to: {results['mesh_obj']}")
        print(f"Depth map saved to: {results['depth_map']}")
        print(f"Visualization saved to: {results['visualization']}")
        
    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    main()