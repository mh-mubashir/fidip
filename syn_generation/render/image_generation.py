'''
Created by Xiaofei Huang (xhuang@ece.neu.edu)
Modified to use PyRender instead of OpenDR
'''
import numpy as np
import os
import cv2
import pyrender
import trimesh
from smil_webuser.serialization import load_model
import random
from pickle import load

# Set up headless rendering for HPC environment
os.environ['PYOPENGL_PLATFORM'] = 'egl'  # Use EGL for headless rendering
os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Disable Qt display
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'  # Enable OpenEXR support

# Force PyRender to use software rendering if needed
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'  # Try OSMesa first (software rendering)
try:
    import pyrender
    # Test if we can create a renderer
    test_renderer = pyrender.OffscreenRenderer(100, 100)
    test_renderer.delete()
    print("PyRender EGL/OSMesa working - using hardware/software rendering")
except:
    print("OSMesa failed, trying EGL...")
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Removed fallback function - using PyRender only

## Only needed for pose prior
class Mahalanobis(object):

    def __init__(self, mean, prec, prefix):
        self.mean = mean
        self.prec = prec
        self.prefix = prefix

    def __call__(self, pose):
        if len(pose.shape) == 1:
            return (pose[self.prefix:]-self.mean).reshape(1, -1).dot(self.prec)
        else:
            return (pose[:, self.prefix:]-self.mean).dot(self.prec)

syn_folder = '../output/synthetic_images'

## Assign attributes to renderer
w, h = (640, 480)

## Load SMIL model
m, kin_table = load_model('smil_web.pkl')
tmpl = trimesh.load('template.obj')


## List background images
bg_folder = 'backgrounds'
bg_list = []                                                                                                            
bg_subdirs = [x[0] for x in os.walk(bg_folder)]                                                                            
for subdir in bg_subdirs:                                                                                            
    files = os.walk(subdir).__next__()[2]                                                                             
    if (len(files) > 0):                                                                                         
        for file in files:                                                                                        
            bg_list.append(subdir + "/" + file)       
#print(bg_list)


## List texture images
txt_folder = 'textures'
txt_list = []                                                                                                            
txt_subdirs = [x[0] for x in os.walk(txt_folder)]                                                                            
for subdir in txt_subdirs:                                                                                            
    files = os.walk(subdir).__next__()[2]                                                                             
    if (len(files) > 0):                                                                                          
        for file in files:                                                                                        
            txt_list.append(subdir + "/" + file)        
#print(txt_list)

num = 0
bodies_folder = '../output/results'
print(f"Looking for body files in: {bodies_folder}")
print(f"Available directories: {[x[0] for x in os.walk(bodies_folder)]}")

for x in os.walk(bodies_folder):  
    if x[0] == bodies_folder:
        continue 
    
    print(f"Processing directory: {x[0]}")
    cur_body_file = os.path.join(x[0], '000.pkl')
    cur_conf_file = os.path.join(x[0], 'conf.yaml')
    
    if not os.path.exists(cur_body_file):
        print(f"Body file not found: {cur_body_file}")
        continue
        
    print(f"Loading body params from: {cur_body_file}")
    body_params = load(open(cur_body_file, 'rb'))
    #print(body_params)

    m.pose[:3] = body_params['global_orient']
    m.pose[3:] = body_params['body_pose']
    m.betas[:] = body_params['betas']
    trans = body_params['camera_translation'][0]

    g_rot0 = float(m.pose[0])
    g_rot1 = float(m.pose[1])          
    g_rot2 = float(m.pose[2])

    for i in range(10):
        num = num + 1
        # syn: change global rotation
        # m.pose[0] = g_rot0 - np.pi/18 * (i)
        # m.pose[1] = g_rot1 + np.pi/18 * (i)
        # m.pose[2] = g_rot2 + np.pi/18 * (i)

        m.pose[0] = g_rot0 + np.pi/11 * (i)
        m.pose[1] = g_rot1 - np.pi/11 * (i)
        m.pose[2] = g_rot2 - np.pi/11 * (i)


        bg_idx = num % len(bg_list)
        bg_file = bg_list[bg_idx]
        bg = cv2.imread(bg_file)
        x = 0
        y = 0
        bg_im = bg[y:y+h, x:x+w].astype(np.float64)/255
      
        txt_idx = num % len(txt_list)
        txt_file = txt_list[txt_idx]
        txt = cv2.imread(txt_file)
        txt = cv2.cvtColor(txt, cv2.COLOR_BGR2RGB)
        txt_im = txt.astype(np.float64)/255 

        # Create PyRender scene
        scene = pyrender.Scene()
        
        # Create mesh from SMIL model
        mesh = trimesh.Trimesh(vertices=m.r, faces=m.f, vertex_colors=np.ones_like(m.r))
        
        # Apply texture if available
        if hasattr(tmpl, 'vt') and hasattr(tmpl, 'ft'):
            # Create texture coordinates
            mesh.visual.uv = tmpl.vt
            # Apply texture image
            if txt_im is not None:
                # Create material with texture
                material = pyrender.MetallicRoughnessMaterial(
                    baseColorTexture=pyrender.Texture(source=txt_im),
                    metallicFactor=0.0,
                    roughnessFactor=0.8
                )
            else:
                material = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[0.8, 0.8, 0.8, 1.0],
                    metallicFactor=0.0,
                    roughnessFactor=0.8
                )
        else:
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.8, 0.8, 0.8, 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.8
            )
        
        # Add mesh to scene
        mesh_node = scene.add(pyrender.Mesh.from_trimesh(mesh, material=material))
        
        # Set up camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi/4.0, aspectRatio=w/h)  # Wider FOV
        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.1],
            [0.0, 0.0, 1.0, 2.0],  # Move camera much further back
            [0.0, 0.0, 0.0, 1.0]
        ])
        scene.add(camera, pose=camera_pose)
        
        # Add lighting
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light)
        
        # Render with PyRender (no fallback)
        renderer = pyrender.OffscreenRenderer(w, h)
        color, depth = renderer.render(scene)
        renderer.delete()  # Clean up renderer
        
        # Composite with background
        if bg_im is not None:
            # Convert background to same size
            bg_resized = cv2.resize(bg_im, (w, h))
            # Create mask from depth
            mask = (depth > 0).astype(np.float32)
            mask = np.stack([mask, mask, mask], axis=2)
            # Composite
            data = (color * mask + bg_resized * (1 - mask) * 255).astype(np.uint8)
        else:
            data = color.astype(np.uint8)
        
        # Skip display on headless systems
        # cv2.imshow('render_SMIL', data)  # Commented out for headless
        
        file_name = 'syn' + str(num) + '.jpg'
        syn_file = os.path.join(syn_folder, file_name)
        cv2.imwrite(syn_file, data)
        print(f"Generated synthetic image: {syn_file}")


        

    





