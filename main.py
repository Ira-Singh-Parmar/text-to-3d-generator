import sys
import os
import torch
import imageio
import trimesh

# -------------------------------
# Add TripoSR folder to Python path
tripo_path = os.path.join(os.path.dirname(__file__), "TripoSR")
if tripo_path not in sys.path:
    sys.path.insert(0, tripo_path)

# -------------------------------
# Headless rendering for pyrender
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import pyrender

# -------------------------------
# Import TSR
from tsr.system import TSR

# -------------------------------
# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -------------------------------
# Text prompt for Stable Diffusion input
prompt = "a futuristic flying motorcycle"

# -------------------------------
# Generate input image using Stable Diffusion
from diffusers import StableDiffusionPipeline

print("Generating input image from text prompt...")
sd = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
).to(device)

image = sd(prompt).images[0]
image.save("input.png")
print("Saved input.png")

# -------------------------------
# Load TSR model
print("Loading TSR model...")

# Make sure you replace the model paths if you have local weights
tsr_model = TSR.from_pretrained(
    pretrained_model_name_or_path="TripoAI/TripoSR",  # HF repo or local folder
    config_name="config.yaml",                        # replace with actual config
    weight_name="model.pth"                           # replace with actual weights
)
tsr_model.configure()  # initialize all modules

# -------------------------------
# Forward pass to get scene codes
print("Encoding image into 3D scene codes...")
scene_codes = tsr_model.forward("input.png", device=device)

# -------------------------------
# Extract 3D mesh
print("Extracting 3D mesh from scene codes...")
meshes = tsr_model.extract_mesh(scene_codes, resolution=128)  # lower for speed
meshes[0].export("output.obj")
print("Saved output.obj")

# -------------------------------
# Render mesh to PNG using pyrender
print("Rendering mesh to PNG...")
scene = pyrender.Scene()
scene.add(pyrender.Mesh.from_trimesh(meshes[0]))
renderer = pyrender.OffscreenRenderer(512, 512)
color, _ = renderer.render(scene)
imageio.imwrite("render.png", color)
renderer.delete()
print("Saved render.png")

print("✅ Finished successfully!")
