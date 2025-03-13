import os
import uuid
import scipy.ndimage
from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from torchvision import transforms
from transformers import ViTModel
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
OUTPUT_FOLDER = os.path.join('static', 'outputs')
MODEL_CACHE_DIR = os.path.join('models', 'cache')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MODEL_CACHE_DIR'] = MODEL_CACHE_DIR

# Hugging Face model configuration
app.config['HF_MODEL_ID'] = 'Ratnesh007/optimized_lrgt_3d_reconstruction'  # Replace with your actual username/model-id
app.config['LOCAL_MODEL_PATH'] = 'models/optimized_lrgt_3d_reconstruction.pth'  # Fallback to local model if HF fails
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Model definition
class Simple3DModel(torch.nn.Module):
    def __init__(self):
        super(Simple3DModel, self).__init__()
        self.encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.fc = torch.nn.Linear(768, 128 * 128 * 128)

    def forward(self, x):
        x = self.encoder(x).last_hidden_state[:, 0, :]
        x = self.fc(x)
        return x.view(-1, 128, 128, 128)


# Load model (lazy loading)
model = None


def load_model():
    global model
    if model is None:
        model = Simple3DModel().to(device)

        try:
            # First try loading from Hugging Face Hub
            print(f"Attempting to load model from Hugging Face: {app.config['HF_MODEL_ID']}")

            from huggingface_hub import snapshot_download
            # Download the model files from HF Hub
            model_dir = snapshot_download(
                repo_id=app.config['HF_MODEL_ID'],
                cache_dir=app.config['MODEL_CACHE_DIR'],
                allow_patterns=["*.pth", "*.bin", "*.json", "*.model", "*.pt"],
                local_files_only=False  # Set to True if you want to use only cached files
            )

            # Find the model file in the downloaded directory
            model_files = [f for f in os.listdir(model_dir) if f.endswith(('.pth', '.pt', '.bin'))]
            if model_files:
                model_path = os.path.join(model_dir, model_files[0])
                print(f"Loading model from Hugging Face: {model_path}")
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                print("Model loaded successfully from Hugging Face Hub")
                return model
            else:
                print("No model weights found in the downloaded files")

        except Exception as e:
            print(f"Error loading model from Hugging Face: {str(e)}")
            print("Falling back to local model...")

        # Fallback to local model if Hugging Face fails
        try:
            local_model_path = app.config['LOCAL_MODEL_PATH']
            if os.path.exists(local_model_path):
                print(f"Loading local model from: {local_model_path}")
                model.load_state_dict(torch.load(local_model_path, map_location=device))
                model.eval()
                print("Local model loaded successfully")
                return model
            else:
                print(f"Error: Local model file not found at {local_model_path}")
                return None
        except Exception as e:
            print(f"Error loading local model: {str(e)}")
            return None

    return model


# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict_3d_from_image(image_path):
    """Generate 3D model from image"""
    try:
        # Load model if not already loaded
        model = load_model()
        if model is None:
            print("Model failed to load")
            return np.zeros((128, 128, 128))  # Return empty voxel grid as fallback

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Generate prediction
        with torch.no_grad():
            predicted_voxel = model(image_tensor).cpu().numpy().squeeze()

        # Print statistics about the voxel data
        print(
            f"Voxel statistics: min={predicted_voxel.min()}, max={predicted_voxel.max()}, mean={predicted_voxel.mean()}")
        print(f"Number of voxels > 0.5: {np.sum(predicted_voxel > 0.5)}")
        print(f"Number of voxels > 0.3: {np.sum(predicted_voxel > 0.3)}")
        print(f"Number of voxels > 0.1: {np.sum(predicted_voxel > 0.1)}")

        return predicted_voxel
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return np.zeros((128, 128, 128))  # Return empty voxel grid


def smooth_voxel_data(voxel_data, threshold=0.3, sigma=1.2):
    """
    Apply Gaussian smoothing to voxel data before thresholding

    Parameters:
        voxel_data: numpy array of voxel predictions
        threshold: value threshold for binary voxel decision (lowered from 0.5 to 0.3)
        sigma: smoothing factor for Gaussian filter

    Returns:
        Binary voxel grid after smoothing and thresholding
    """
    # Print statistics about the voxel data
    print(f"Voxel statistics: min={voxel_data.min()}, max={voxel_data.max()}, mean={voxel_data.mean()}")
    print(f"Number of voxels > 0.5: {np.sum(voxel_data > 0.5)}")
    print(f"Number of voxels > 0.3: {np.sum(voxel_data > 0.3)}")
    print(f"Number of voxels > 0.1: {np.sum(voxel_data > 0.1)}")

    # Apply Gaussian smoothing to the raw voxel predictions
    smoothed_data = scipy.ndimage.gaussian_filter(voxel_data, sigma=sigma)

    # Apply threshold after smoothing
    binary_voxels = smoothed_data > threshold

    return binary_voxels


def save_as_smooth_mesh(voxel_data, filename, threshold=0.3, sigma=1.2,
                        depth=9, scale=1.1, linear_fit=False):
    """
    Convert voxel data to a smooth mesh using Poisson surface reconstruction

    Parameters:
        voxel_data: numpy array of voxel predictions
        filename: output filename
        threshold: value threshold for binary voxel decision (lowered from 0.5 to 0.3)
        sigma: smoothing factor for Gaussian filter
        depth: depth parameter for Poisson reconstruction (higher = more detail)
        scale: scale factor for the reconstructed mesh
        linear_fit: whether to use linear fit for color interpolation

    Returns:
        Tuple of (success_bool, message_string)
    """
    try:
        # Apply smoothing to the voxel data
        binary_voxels = smooth_voxel_data(voxel_data, threshold, sigma)

        # Extract voxel indices where value > threshold
        voxel_indices = np.argwhere(binary_voxels)

        print(f"Found {len(voxel_indices)} voxels after thresholding at {threshold}")

        # If no voxels above threshold, try with a lower threshold
        if voxel_indices.size == 0:
            print(f"No voxels above threshold {threshold}, trying with lower threshold 0.1")
            new_threshold = 0.1
            binary_voxels = smooth_voxel_data(voxel_data, new_threshold, sigma)
            voxel_indices = np.argwhere(binary_voxels)

            if voxel_indices.size == 0:
                print("No voxels detected after thresholding. Creating fallback model.")
                # Create a simple cube as fallback
                mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
                mesh.compute_vertex_normals()
                o3d.io.write_triangle_mesh(filename, mesh, write_vertex_normals=True)
                return False, "No clear 3D structure detected. A simple placeholder has been created."

        # Convert voxel indices to point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(voxel_indices.astype(np.float32))

        # Estimate normals with consistent orientation
        # Increased parameters for better normal estimation
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=50))
        pcd.orient_normals_consistent_tangent_plane(k=30)

        # Apply Poisson surface reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, scale=scale, linear_fit=linear_fit)

        if mesh.is_empty():
            print("Poisson reconstruction failed. Creating fallback mesh.")
            mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
            mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(filename, mesh, write_vertex_normals=True)
            return False, "3D reconstruction failed. The image may not contain recognizable 3D structure."

        # Remove low-density vertices (outliers)
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)

        # Final mesh cleanup and preparation
        mesh.compute_vertex_normals()

        # Apply Laplacian smoothing for even smoother results
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=5)

        # Save the final mesh
        o3d.io.write_triangle_mesh(filename, mesh, write_vertex_normals=True)
        return True, "3D model generated successfully with advanced smoothing techniques."

    except Exception as e:
        print(f"Error in smooth mesh generation: {str(e)}")
        # Create a simple fallback mesh
        try:
            mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
            mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(filename, mesh, write_vertex_normals=True)
        except Exception as inner_e:
            print(f"Error creating fallback mesh: {str(inner_e)}")
        return False, f"Error in 3D generation: {str(e)}"


def visualize_voxel_grid(voxel_data, threshold=0.3):
    """
    Utility function for debugging - creates a point cloud visualization of voxel data
    """
    voxel_data = (voxel_data > threshold).astype(np.uint8)  # Lower threshold
    voxel_indices = np.argwhere(voxel_data)

    # Check if we have any points before creating point cloud
    if len(voxel_indices) == 0:
        print(f"No voxels detected with threshold {threshold}. Try lowering the threshold value.")
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_indices)
    return pcd


def save_as_obj(voxel_data, filename):
    """
    Legacy method - Convert voxel data to OBJ file using ball pivoting
    Now serves as a fallback if the smooth mesh generation fails
    """
    # Try with lower threshold of 0.3 instead of 0.5
    voxel_indices = np.argwhere(voxel_data > 0.3)

    if voxel_indices.size == 0:
        # Try even lower threshold
        voxel_indices = np.argwhere(voxel_data > 0.1)

    if voxel_indices.size == 0:
        print("No voxels detected, using fallback option.")
        # Create a simple cube as fallback
        mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(filename, mesh, write_vertex_normals=True)
        return True, "Simple 3D model generated (fallback)."

    # Convert voxel indices to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_indices.astype(np.float32))

    # Compute normals with improved parameters
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=50))
    pcd.orient_normals_consistent_tangent_plane(k=30)

    # Apply Ball Pivoting Algorithm
    radii = [1, 2, 3]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )

    if mesh.is_empty():
        print("Ball Pivoting failed, creating fallback mesh")
        mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(filename, mesh, write_vertex_normals=True)
        return False, "Mesh generation failed. The image may not contain recognizable 3D structure."

    # Apply Laplacian smoothing for better results
    mesh = mesh.filter_smooth_laplacian(number_of_iterations=5)

    # Compute vertex normals
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # Save mesh as OBJ file
    o3d.io.write_triangle_mesh(filename, mesh, write_vertex_normals=True)
    return True, "3D model generated successfully."


# Define endpoint for model info
@app.route('/model-info')
def model_info():
    """Return information about the currently loaded model"""
    try:
        # Check if model is loaded
        global model
        if model is None:
            load_model()

        if model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model failed to load'
            })

        # Get model information
        return jsonify({
            'status': 'success',
            'model_type': 'Simple3DModel',
            'source': app.config['HF_MODEL_ID'],
            'device': str(device),
            'encoder': 'ViT-Base'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Create unique filename
        unique_filename = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        image_filename = unique_filename + file_extension
        obj_filename = unique_filename + '.obj'

        # Save paths
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        obj_path = os.path.join(app.config['OUTPUT_FOLDER'], obj_filename)

        # Save uploaded image
        file.save(image_path)

        try:
            # Process image to generate 3D voxel data
            voxel_data = predict_3d_from_image(image_path)

            # Get mesh generation quality parameter from request (default to high quality)
            quality = request.form.get('quality', 'high')

            if quality == 'low':
                # Use legacy method for quick results
                success, message = save_as_obj(voxel_data, obj_path)
            else:
                # Use advanced smooth mesh generation (higher quality but slower)
                # Updated to use lower threshold of 0.3 instead of 0.45
                success, message = save_as_smooth_mesh(
                    voxel_data,
                    obj_path,
                    threshold=0.3,  # Lower threshold to capture more detail
                    sigma=1.2,  # Smoothing parameter
                    depth=9,  # Poisson reconstruction detail level
                    scale=1.1  # Scale factor
                )

            # Return paths for frontend to use along with success message
            return jsonify({
                'image': image_path,
                'obj': os.path.join('static', 'outputs', obj_filename),
                'download_url': f'/download/{obj_filename}',
                'message': message,
                'success': success
            })

        except Exception as e:
            print(f"Error: {str(e)}")
            return jsonify({'error': 'Failed to process image'}), 500

    return jsonify({'error': 'Invalid file'}), 400


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)