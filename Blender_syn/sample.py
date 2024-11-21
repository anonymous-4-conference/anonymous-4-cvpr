import bpy
import numpy as np
from mathutils import  Vector, Euler
import sys
import os
import yaml
# import argparse
import numpy as np
import csv
import cv2
import open3d as o3d
import pdb 
from tqdm import tqdm
from skimage import measure
import pandas as pd
import shutil
parent_dir =os.path.dirname( os.path.abspath(__file__))
sys.path.append(parent_dir)
from crystal_utils import *
from itertools import product
import bpy
import random
""" install requirements"""
np.random.seed(99)
# blender -b --python-use-system-env --python-console
#blender -b -P sample.py --python-use-system-env

# import subprocess
# import sys

# python_executable = sys.executable
# subprocess.call([python_executable, '-m', 'ensurepip'])
# subprocess.call([python_executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
# subprocess.call([python_executable, '-m', 'pip', 'install', 'scikit-image'])
# subprocess.call([python_executable, '-m', 'pip', 'install', 'open3d'])
# subprocess.call([python_executable, '-m', 'pip', 'install', 'opencv-python'])
# subprocess.call([python_executable, '-m', 'pip', 'install', 'pandas'])
# # Command to be executed with administrative privileges
# command = "cmd"

# # Using 'runas' to execute the command as an administrator
# subprocess.run(f"runas /user:administrator {command}", shell=True)
# python_exe = sys.executable
# subprocess.call([python_exe, "-m", "ensurepip"])
# subprocess.call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
# subprocess.call(['runas', '/user:administrator',python_exe, "-m", "pip", "install", "scikit-image","--user"])
""" install requirements"""

""" tutorials"""
# remove overlapping areas
# https://www.youtube.com/watch?v=a0SQarpKzYg

## fluid simulation
    # basic formation
    # https://www.youtube.com/watch?v=e3mhJXuveFo
    # sticky on the crystal
    # https://www.youtube.com/watch?v=Q7sKdxMlNks
""" tutorials"""


def loading_obj(obj_file_path):

        bpy.ops.import_scene.obj(filepath=obj_file_path)
        loop_obj = bpy.context.view_layer.objects.active
        # bpy.context.view_layer.objects.active = loop_obj
        # bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        return loop_obj

def loading_dae(dae_file_path):
    bpy.ops.wm.collada_import(filepath=dae_file_path)
    loop_obj = bpy.context.view_layer.objects.active
    return loop_obj



def polygonize_loop(config,loop_pth):

    py_dir=config['loop']['preprocess']['py_pth']
    # save_dir =os.path.dirname(py_dir)
    save_dir =loop_pth
    array = np.load(py_dir).astype(np.uint8)

    kernel = np.ones((5,5), np.uint8)
    processed_array = np.empty_like(array)


    for i in tqdm(range(array.shape[2]), desc="Erosion-Dilation denoising loop"):
        # Apply dilation
        tmp = array[:,:,i]
        eroded = cv2.erode(tmp, kernel)
        dilated = cv2.dilate(eroded, kernel)

        # Store the processed image
        processed_array[:,:,i] = dilated
            
    # np.save(save_dir + '/processed_loop_.npy', processed_array)

    verts, faces, _, _ = measure.marching_cubes(processed_array == 2, level=0, spacing=(1.0, 1.0, 1.0),)

        # verts=np.load(loop_pth + '/loop_verts.npy')
        # faces=np.load(loop_pth + '/loop_faces.npy')

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    decimation_ratio = config['loop']['preprocess']['decimation_ratio']
    decimated_mesh = mesh.simplify_quadric_decimation(
        target_number_of_triangles=int(len(mesh.triangles) * decimation_ratio)
    )
    decimated_verts = np.asarray(decimated_mesh.vertices)
    decimated_faces = np.asarray(decimated_mesh.triangles)
    np.save(save_dir + '/loop_verts_deci_{}.npy'.format(decimation_ratio), decimated_verts)
    np.save(save_dir + '/loop_faces_deci_{}.npy'.format(decimation_ratio), decimated_faces)


def create_loop(config,loop_pth,save_obj = True):
    
    if config['loop']['is_preprocess']:
        polygonize_loop(config,loop_pth)
   
    list_scale_x = config['loop']['params']['list_scale_x']
    list_scale_y = config['loop']['params']['list_scale_y']
    list_scale_z = config['loop']['params']['list_scale_z']


    # center_world = loop_obj.matrix_world @ Vector(center_loop)
    # loop_obj.location -= center_world   

    for scale_x in list_scale_x:
        for scale_y in list_scale_y:
            for scale_z in list_scale_z:


                verts = np.load(loop_pth + '/loop_verts_deci_{}.npy'.format(config['loop']['preprocess']['decimation_ratio']))
                faces = np.load(loop_pth + '/loop_faces_deci_{}.npy'.format(config['loop']['preprocess']['decimation_ratio']))
                loop_mesh = bpy.data.meshes.new("LoopMesh")

                loop_mesh.from_pydata(verts.tolist(), [], faces.tolist())
                loop_mesh.update()
                # Create a new object with the mesh and link it to the current collection
                loop_obj = bpy.data.objects.new("LoopObject", loop_mesh)
                bpy.context.collection.objects.link(loop_obj)

                # Set the created object as active and select it
                bpy.context.view_layer.objects.active = loop_obj
                loop_obj.select_set(True)  # Replace with your loop object's name

                angle_in_radians_y = np.radians(-90)
                angle_in_radians_z = np.radians(-30)
                loop_obj.rotation_euler = Euler((0, angle_in_radians_y,angle_in_radians_z), 'XYZ')
                bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

                min_bound = np.array(loop_obj.bound_box[0])
                max_bound = np.array(loop_obj.bound_box[6])
                center_loop = - (min_bound + max_bound) / 2

                bpy.context.view_layer.update()
                bpy.ops.transform.translate(value=Vector(center_loop))
                bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

                loop_obj.scale = scale_x, scale_y, scale_z
                bpy.context.view_layer.update()
                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
                # obj_name = "./loop_o3d_{}_{}_{}_{}.obj".format(config['loop']['preprocess']['decimation_ratio'],scale_x, scale_y, scale_z)
                obj_name = "./loop_o3d_{}_{}_{}_{}.dae".format(config['loop']['preprocess']['decimation_ratio'],scale_x, scale_y, scale_z)
                bpy.ops.object.select_all(action='DESELECT') 
                loop_obj.name =obj_name.replace('.dae','')
                loop_obj.select_set(True)
                bpy.context.view_layer.objects.active = loop_obj  
                file_path=loop_pth+ obj_name
                bpy.ops.export_scene.obj(filepath=file_path, use_selection=True)
                bpy.ops.wm.collada_export(filepath=file_path, selected=True)
                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete() 

def create_crystal(config,loop_pth,crystal_pth,crystal_writer,crystal_writer_file,crystal_dict):       
    # Calculate the center of the loop
    # loop_obj=loading_obj(os.path.join(loop_pth,'loop_o3d_0.001_1_1_1.obj'))
    # loop_obj =loading_dae(os.path.join(loop_pth,'loop_o3d_0.001_1_1_1.dae'))
    
   
    # loop_min_bound = np.array(loop_obj.bound_box[0])
    # loop_max_bound = np.array(loop_obj.bound_box[6])
    
    # center_loop = (loop_min_bound + loop_max_bound) / 2   
    # loop_dim = [loop_min_bound,loop_max_bound]
    
    param_combinations = product(
        config['crystal']['params']['num_vertices_list'],
        config['crystal']['params']['height_list'],
        config['crystal']['params']['bottom_width_list'],
        config['crystal']['params']['middle_width_list'],
        config['crystal']['params']['top_width_in_percent_list'],
        config['crystal']['params']['top_height_in_percent_list'],
        config['crystal']['params']['scale_factor_list']
    )
    count = 0
    for i,combination in enumerate(param_combinations):
            top_width = combination[4]
            top_height = combination[5]
            bottom_width = combination[2]
            middle_width = combination[3]
            if top_width >= top_height:
                continue
            if middle_width==bottom_width:
                continue
            loop_obj =loading_dae(os.path.join(loop_pth,config['loop']['simiulate_loop']))
            row_dict = crystal_dict.copy()
            
            store_dir,rotation_tuple=create_crystal_single(
            crystal_pth,
            top_width_in_percent=combination[4],
            top_height_in_percent=combination[5],
            vertices=combination[0],
            height=combination[1],
            bottom_width=combination[2],
            middle_width=combination[3],
            scale_factor=combination[6],
            loop_obj=loop_obj,)

            row_dict['crystal_id'] = 'id{:06d}'.format(count)
            row_dict['num_vertices'] = combination[0]
            row_dict['height'] = combination[1]
            row_dict['bottom_width'] = combination[2]
            row_dict['middle_width'] = combination[3]
            row_dict['top_width_in_percent'] = combination[4]
            row_dict['top_height_in_percent'] = combination[5]
            row_dict['scale_factor'] = combination[6]
            row_dict['crystal_path'] = store_dir
            row_dict['rotation_tuple'] = rotation_tuple
            crystal_writer.writerow(row_dict)
            count += 1
            crystal_writer_file.flush()
    # for num_vertices in config['crystal']['params']['num_vertices_list']:
    #     for height in config['crystal']['params']['height_list']:
    #         for bottom_width in config['crystal']['params']['bottom_width_list']:
    #             for middle_width in config['crystal']['params']['middle_width_list']:
    #                 for top_width_in_percent in config['crystal']['params']['top_width_in_percent_list']:
    #                     for top_height_in_percent in config['crystal']['params']['top_height_in_percent_list']:
    #                         for scale_factor in config['crystal']['params']['scale_factor_list']:
                                # create_crystal_single(crystal_pth,top_width_in_percent=top_width_in_percent,top_height_in_percent=top_height_in_percent,vertices = num_vertices ,height = height ,bottom_width =bottom_width ,middle_width = middle_width ,scale_factor =scale_factor,save_obj=True,loop_dim=loop_dim,crystal_writer=crystal_writer,crystal_writer_file=crystal_writer_file,crystal_dict=crystal_dict)

def detect_overlap(loop_obj_name, crystal_obj_name, merge_distance=0.1):
    # Ensure both objects exist in the scene
    if loop_obj_name not in bpy.data.objects or crystal_obj_name not in bpy.data.objects:
        print("One or both objects not found in the scene.")
        return False
    
    # Retrieve the objects based on their names
    original_loop_obj = bpy.data.objects[loop_obj_name]
    original_crystal_obj = bpy.data.objects[crystal_obj_name]

    # Duplicate the objects and make sure the duplicates are not linked
    loop_obj = original_loop_obj.copy()
    loop_obj.data = original_loop_obj.data.copy()
    loop_obj.animation_data_clear()
    bpy.context.collection.objects.link(loop_obj)

    crystal_obj = original_crystal_obj.copy()
    crystal_obj.data = original_crystal_obj.data.copy()
    crystal_obj.animation_data_clear()
    bpy.context.collection.objects.link(crystal_obj)

    # Deselect all objects
    bpy.ops.object.select_all(action='DESELECT')

    # Prepare loop object by selecting and storing vertices
    bpy.context.view_layer.objects.active = loop_obj
    loop_obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_linked()
    loop_vertices = [v.index for v in loop_obj.data.vertices if v.select]
    bpy.ops.object.mode_set(mode='OBJECT')

    # Prepare crystal object by selecting and storing vertices
    bpy.context.view_layer.objects.active = crystal_obj
    crystal_obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_linked()
    crystal_vertices = [v.index for v in crystal_obj.data.vertices if v.select]
    bpy.ops.object.mode_set(mode='OBJECT')

    # Temporarily join objects for merging vertices
    bpy.ops.object.select_all(action='DESELECT')
    loop_obj.select_set(True)
    crystal_obj.select_set(True)
    bpy.context.view_layer.objects.active = loop_obj
    bpy.ops.object.join()

    # Remove doubles to detect overlap
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=merge_distance)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Check for overlap based on vertex count changes
    merged_vertices_count = len(loop_obj.data.vertices)
    overlap = merged_vertices_count < len(loop_vertices) + len(crystal_vertices)

    # Clean up: delete the temporary object
    bpy.ops.object.select_all(action='DESELECT')
    loop_obj.select_set(True)
    bpy.ops.object.delete()

    return overlap

def create_crystal_single(crystal_pth,vertices ,height ,bottom_width ,middle_width,top_width_in_percent,top_height_in_percent,scale_factor = 10 ,loop_obj=None):

    loop_max_bound =get_bounds(loop_obj)[6]
    loop_min_bound =get_bounds(loop_obj)[0]
    bpy.context.scene.vertices = vertices  # Number of vertices
    bpy.context.scene.height = height # Height of the crystal
    bpy.context.scene.bottom_width = bottom_width  # Bottom width
    bpy.context.scene.middle_width =middle_width# Middle width
    bpy.context.scene.top_width_in_percent = top_width_in_percent # Top width in percent
    bpy.context.scene.top_height_in_percent = top_height_in_percent  # Top height in percent
    bpy.context.scene.rot_z=0
    bpy.context.scene.rot_y=0
    bpy.context.scene.rot_x=0
    bpy.ops.crystalsgenerator.create()
    # Get the crystal object (assuming it is the last created object)
    crystal_obj = bpy.context.selected_objects[0]
    
    rot_x = np.radians(-90)  # Rotation angle in radians
    # rot_y = np.radians(-90)
    # rot_y = np.radians(0)

    crystal_obj.scale *= scale_factor
    bpy.context.view_layer.update()
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


    np.random.seed(99)
    rot_x = np.radians(np.random.randint(0,360)) 
    # rot_x = 0
    rot_y = np.radians(np.random.randint(0,180)) 
    rot_z = 0
    rotation_tuple = (rot_x,rot_y,rot_z)
    bpy.ops.transform.rotate(value=rot_x, orient_axis='X', orient_type='GLOBAL')
    bpy.ops.transform.rotate(value=rot_y, orient_axis='Y', orient_type='GLOBAL')
    bpy.ops.transform.rotate(value=rot_z, orient_axis='Z', orient_type='GLOBAL')    
    
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    


    
    # print('name is',name)
    

    # obj1_center = np.mean([np.array(corner) for corner in loop_obj.bound_box], axis=0)
    min_bound = np.array(crystal_obj.bound_box[0])
    max_bound = np.array(crystal_obj.bound_box[6])
    center_crystal = - (min_bound + max_bound) / 2
    center_loop = (loop_max_bound - loop_min_bound) / 2  
    #  
 
    # trans_x = np.random.randint(min_bound[0]+loop_min_bound[0],max_bound[0]+loop_max_bound[0])
    # try:
    #     trans_x = np.random.randint(0,max_bound[0]/2)

    # except:
    #     trans_x= np.random.randint(0,min_bound[0]/2)
    # # trans_x = np.random.randint(loop_min_bound[0],0)
    # try:
    #     trans_y = np.random.randint(0,max_bound[1]/2)
    # except:
    #     trans_y= np.random.randint(0,min_bound[1]/2)
        
    # try:
    #     trans_z = np.random.randint(0,max_bound[2]/2)
    # except:
    #     trans_z= np.random.randint(0,min_bound[2]/2)
    trans_x=0
    trans_y=0
    trans_z=0
    # pdb.set_trace()
    # center_loop[0] += (trans_x +max_bound[0])  # make sure the crystal is certainly above the loop
    center_crystal[0] += trans_x  
    center_crystal[1] += trans_y
    center_crystal[2] += trans_z
    bpy.context.view_layer.update()
    bpy.ops.transform.translate(value=Vector(center_crystal))

    crystal_obj.matrix_world @ Vector(-center_crystal) # move the centre of the crystal to current translated local origin
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
    
    name= "crystal_{}_{}_{}_{}_{}_{}.dae".format(vertices,height,bottom_width,middle_width,top_width_in_percent,top_height_in_percent)
    crystal_obj.name =name.replace('.dae','')   
    loop_obj.name="Loop"
    # pdb.set_trace()
    overlap = detect_overlap(loop_obj.name, crystal_obj.name)
    if overlap:
        print(f" !!!!! Overlap detected on {crystal_obj.name}, adjusting position. !!!!! ")
        pdb.set_trace()
        
    else:
        print(f" No overlap detected on {crystal_obj.name}.")
   
    bpy.ops.object.select_all(action='DESELECT') 
    crystal_obj.select_set(True)
    bpy.context.view_layer.objects.active = crystal_obj  
    # name= "crystal_{}_{}_{}_{}_{}_{}.obj".format(vertices,height,bottom_width,middle_width,top_width_in_percent,top_height_in_percent)
    # crystal_obj.data.name =name.replace('.obj','')
    
    
        
    file_path=os.path.join(crystal_pth,name)

    bpy.ops.wm.collada_export(filepath=file_path, selected=True)
    bpy.ops.object.select_all(action='SELECT') 
    bpy.ops.object.delete()
    return file_path,rotation_tuple

def get_bounds(obj):
    """
    Calculate the world-space bounds of a Blender object.
    """
    local_bounds = [Vector(bound) for bound in obj.bound_box]
    world_bounds = [obj.matrix_world @ bound for bound in local_bounds]
    return np.array([bound.to_tuple() for bound in world_bounds])

def create_liquor_scale(loop_obj,crystal_obj, liquor_diameter, liquor_path,save_obj=False):
    # Duplicate objects
    loop_obj.select_set(True)
    crystal_obj.select_set(True)
    bpy.ops.object.duplicate(linked=False)
    bpy.ops.object.join()  # Joined duplicate
    combined_obj = bpy.context.active_object
    # combined_obj = loop_obj
    combined_obj.name = 'CombinedMesh' # Now loop_duplicate contains both meshes

    bpy.ops.mesh.primitive_uv_sphere_add(radius=liquor_diameter/2, location=(0, 0, 0))
    liquor_obj = bpy.context.active_object
    liquor_obj.name = 'MotherLiquor'

    # Adjust the size of the sphere to be slightly larger than the combined mesh
    liquor_obj.dimensions = combined_obj.dimensions * 1.05 

    shrinkwrap = liquor_obj.modifiers.new(name="Shrinkwrap", type='SHRINKWRAP')
    shrinkwrap.target = combined_obj
    shrinkwrap.offset = 0.01
    # shrinkwrap.wrap_method= 'TARGET_PROJECT'
    #https://docs.blender.org/api/current/bpy.types.ShrinkwrapModifier.html#bpy.types.ShrinkwrapModifier
    if save_obj:
        bpy.ops.object.select_all(action='DESELECT')  
        liquor_obj.select_set(True)
        file_path=loop_pth+ "./liquor_scale.obj"
        bpy.ops.export_scene.obj(filepath=file_path, use_selection=True)
        
    return liquor_obj

def local_to_world(local_coords, world_matrix):
    # Convert the local coordinates to a Vector, then multiply with the world_matrix
    world_coords = world_matrix @ Vector(local_coords)
    return world_coords

def create_liquor_fluid(config,loop_pth, crystal_paths,crystal_id,liquor_path,liquor_writer,liquor_writer_file,liquor_dict):
    # loop_obj =loading_dae(os.path.join(loop_pth,config['loop']['simiulate_loop']))
    
    
    for i, crystal_pth in enumerate(crystal_paths):
        # crystal_obj =loading_dae(crystal_pth)
        row_dict = liquor_dict.copy()
        id = 'id{:06d}'.format(i)
        # try:
        
        # crystal_pth=crystal_pth.replace("crystal_3_8_1_1_0_25.dae","crystal_3_8_1_2_100_75.dae")
        
        file_path,frame,liquor_shape = create_liquor_fluid_single(loop_pth, crystal_pth,config,liquor_path,id)
        # 
        # except:
        #     print('error in creating liquor for {}'.format(crystal_pth))
        #      
        # if file_path is None:
        #     continue
        row_dict['Data_id'] = id
        row_dict['crystal_id'] = crystal_id[i]
        row_dict['overall_path'] = file_path
        row_dict['diffusion'] = config['liquor']['params']['diffusion']
        row_dict['surface_tension'] = config['liquor']['params']['surface_tension']
        row_dict['frame'] = frame
        row_dict['liquor_shape'] = liquor_shape
        liquor_writer.writerow(row_dict)
        liquor_writer_file.flush()
        if frame is not None:
            shutil.rmtree("./cache_{}".format(id))
        # exit()
       
def create_liquor_fluid_single(loop_pth, crystal_pth,config,liquor_path,id):
    name = "liquor_{}_{}_{}.dae".format(config['liquor']['params']['diffusion'],config['liquor']['params']['surface_tension'],id)
    name = "liquor_{}".format(os.path.basename(crystal_pth))
    file_path=os.path.join(liquor_path,name)
    if os.path.exists(file_path):
        print('{} is already created'.format(file_path))
        return file_path, None,None
    loop_obj =loading_dae(os.path.join(loop_pth,config['loop']['simiulate_loop']))
    
    crystal_obj =loading_dae(crystal_pth)
    loop_obj_name = loop_obj.name
    crystal_obj_name = crystal_obj.name
    all_bounds = []
    for obj_name in [ loop_obj_name,crystal_obj_name]:
        obj = bpy.data.objects[obj_name]
        obj_bounds = get_bounds(obj)
        all_bounds.append(obj_bounds)
    
    crystal_max_bound=np.array(get_bounds(crystal_obj)[6])
    crystal_min_bound=np.array(get_bounds(crystal_obj)[0])
    center_crystal = (crystal_max_bound + crystal_min_bound) / 2


    loop_max_bound = np.array(get_bounds(loop_obj)[6])
    loop_min_bound = np.array(get_bounds(loop_obj)[0])
    center_loop = (loop_max_bound + loop_min_bound) / 2
    # Find the min and max bounds
    all_bounds = np.vstack(all_bounds)
    min_bound = np.min(all_bounds, axis=0)
 
    max_bound = np.max(all_bounds, axis=0)
    # crystal_min_bound = np.min( get_bounds(bpy.data.objects[crystal_obj_name]), axis=0)
    # crystal_max_bound = np.max( get_bounds(bpy.data.objects[crystal_obj_name]), axis=0)
    # loop_min_bound = np.min( get_bounds(bpy.data.objects[loop_obj_name]), axis=0)
    # loop_max_bound = np.max( get_bounds(bpy.data.objects[loop_obj_name]), axis=0)
    diff_bound = max_bound - min_bound
    
    bpy.context.scene.gravity[0] = -9.81  # Gravity in X
    bpy.context.scene.gravity[1] = 0     # Gravity in Y
    bpy.context.scene.gravity[2] = 0     # Gravity in Z
    bpy.ops.mesh.primitive_cube_add()
    domain_obj = bpy.context.active_object
    domain_obj.name = 'FluidDomain'
    # domain_obj.scale = 500,500,500
    domain_obj.scale = diff_bound[0]*3,diff_bound[1],diff_bound[2] # the diffusion happens along the x axis
    # Set domain physics properties for fluid simulation
    domain_obj.modifiers.new(name='FluidDomainModifier', type='FLUID')

    domain_obj.modifiers['FluidDomainModifier'].fluid_type = 'DOMAIN'
    domain_settings = domain_obj.modifiers['FluidDomainModifier'].domain_settings
    domain_settings.domain_type = 'LIQUID'
    domain_settings.cache_directory = "cache_{}".format(id)
    # domain_settings.use_adaptive_time_steps = True
    domain_settings.resolution_max =32 
    domain_settings.use_diffusion = True
    if config['liquor']['params']['diffusion']=='low':
        domain_settings.viscosity_base = 1 
        domain_settings.viscosity_exponent = 6
    elif config['liquor']['params']['diffusion']=='high':
        domain_settings.viscosity_base = 2
        domain_settings.viscosity_exponent = 3
    elif config['liquor']['params']['diffusion']=='medium':
        domain_settings.viscosity_base = 5
        domain_settings.viscosity_exponent = 5
    domain_settings.cache_type = 'ALL'
    domain_settings.use_mesh=True
    domain_settings.mesh_scale=2
    domain_settings.surface_tension=config['liquor']['params']['surface_tension']


    # Create a fluid source object (e.g., a sphere)
    fluid_scale_x,fluid_scale_y,fluid_scale_z=config['liquor']['params']['fluid_scale']
    bpy.ops.mesh.primitive_uv_sphere_add(location=(0, 0, 0))
    fluid_obj = bpy.context.active_object
    fluid_obj.name = 'source'
    # fluid_obj.scale = 100,300,400
 
    ran_x = np.random.uniform(0.4,0.6)
    # ran_x=0.35
    # ran_x = 0.4
    ran_y =np. random.uniform(0.9, 1.1)
    # ran_y=1.1
    # ran_y=0.85
    ran_z = np.random.uniform(0.9, 1.0)
    # ran_z = 1.1
    print('')
    print('ran_x,ran_y,ran_z is')
    print(ran_x,ran_y,ran_z)
    print('')
    liquor_shape=(ran_x,ran_y,ran_z)
    # ran_z =0.8
    fluid_obj.scale = diff_bound[0]*fluid_scale_x *ran_x,diff_bound[1]*fluid_scale_y/2 * ran_y,diff_bound[1]*fluid_scale_z/2*ran_z
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # fluid_translate_x =crystal_max_bound[0]/3+ diff_bound[0]*fluid_scale_x
    # fluid_translate_x =loop_max_bound[0]/3+ diff_bound[0]*fluid_scale_x
    # inner diametter 380, outer 500
    fluid_translate_x =-center_loop[0] 
    fluid_translate_z =-center_loop[2] -0
    print('fluid_translate_x,fluid_translate_z is')
    print(fluid_translate_x,fluid_translate_z)
    # pdb.set_trace() 
    bpy.ops.transform.translate(value=Vector((fluid_translate_x,0,fluid_translate_z)))
    
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
    # Set fluid source properties
    fluid_obj.modifiers.new(name='FluidSourceModifier', type='FLUID')
    fluid_obj.modifiers['FluidSourceModifier'].fluid_type = 'FLOW'
    
    flow_settings = fluid_obj.modifiers['FluidSourceModifier'].flow_settings
    flow_settings.flow_type = 'LIQUID'
    flow_settings.flow_behavior = "GEOMETRY"
    bpy.context.view_layer.objects.active = domain_obj


    for obj_name in [loop_obj_name, crystal_obj_name]:
        obj = bpy.data.objects[obj_name]
        obj.modifiers.new(name='Fluid', type='FLUID')
        obj.modifiers['Fluid'].fluid_type = 'EFFECTOR'
        # effector_settings = obj.modifiers['Fluid'].effector_settings
        # effector_settings.surface_distance = 1  
    # 
    # # crystal_obj = bpy.data.objects[crystal_obj_name]
    # # loop_obj = bpy.data.objects[loop_obj_name]
    #  
    # blender_file =file_path
    #blender_file = file_path.replace('.dae','.blend')
    #bpy.ops.wm.save_as_mainfile(filepath=blender_file)
    #  
 
    bpy.ops.fluid.bake_all()
    
    desired_frame_factor =config['liquor']['params']['frame_factor'] # Replace with the frame you want to export
    fluid_max_bound =  get_bounds(fluid_obj)[6]
    fluid_min_bound = get_bounds(fluid_obj)[0]
    
    # world_matrix = domain_obj.matrix_world
    # world_bound_box = [local_to_world(corner, domain_obj.matrix_world) for corner in domain_obj.bound_box]

    # domain_obj.bound_box = world_bound_box
    center_fluid = (fluid_max_bound + fluid_min_bound) / 2  
    distance = center_fluid[0] - crystal_max_bound[0]
    frame =np.sqrt(np.abs(distance*2/bpy.context.scene.gravity[0])) *desired_frame_factor
    frame=random.randint(1,10)
    bpy.context.scene.frame_set(frame)
    bpy.ops.object.modifier_apply(modifier='FluidDomainModifier')
    # Ensure the domain object is selected
    bpy.ops.object.select_all(action='DESELECT')
    domain_obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    #  
    
    # pdb.set_trace()
    domain_obj.data.name =name.replace('.dae','')  
    bpy.ops.wm.save_as_mainfile(filepath=file_path.replace('.dae','.blend'))
    bpy.ops.wm.collada_export(filepath=file_path, selected=True)
    bpy.ops.object.select_all(action='SELECT') 
    bpy.ops.object.delete()
    # 

    # cache_path = domain_obj.modifiers["Fluid"].domain_settings.cache_directory
    # if os.path.exists(cache_path):

    #     shutil.rmtree(cache_path)
    return file_path,frame,liquor_shape


def create_csv_writer(csv_filename, fieldnames):
    file = open(csv_filename, 'w+', newline='')
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    return writer, file

if __name__ == "__main__":
    addon_name = "main"  # install crystal generator
    #  
    # Enable the add-on
    bpy.ops.preferences.addon_enable(module=addon_name)

    global metadata_template
    # there are 3 parts of the crystal, top surface, middle part/area/surface and bottom surface
    metadata_template = {
    'SampleID': None,
    'crystal.num_vertices': None,
    'crystal.height': None, # overall hegiht of the crystal
    'crystal.bottom_width': None,
    # the width of the bottom surface, increase this also increase the width of the middle surface (middle_width >>), then increase top surface (top_width >> ), 
    # if bottom_width is smaller than 1 , then it will be set as 1
    # if want a very smaller bottom surface, set the scael_factor as a very small values
    'crystal.middle_width': None, # the width of the middle part/surface, increase this also increase the width of the top surface (top_width >>)
    # if middle_width is smaller than 1 , then it will be set as 1
    'crystal.top_width_in_percent': None, # the width of the area of the top surface 
    'crystal.top_height_in_percent_list': None, # height of the top part after the middle point
    'crystal.scale_factor': None,
    'loop.list_scale_x': None,
    'loop.list_scale_y': None,
    'loop.list_scale_z': None,
    'liquor.fluid_scale': None,
    'liquor.diffusion': None,
    'liquor.surface_tension': None,
    'liquor.simulation_frame': None,
    }
    global config
    with open('./config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    

    base_root = config['setting']['base_root']
    loop_pth =os.path.join(base_root,'Loop_models') # 'D:/lys/studystudy/phd/0-Project_absorption_correction/dataset/Blender'
    crystal_pth =os.path.join(base_root,'Crystal_models') # 'D:/lys/studystudy/phd/0-Project_absorption_correction/dataset/Blender'
    liquor_pth =os.path.join(base_root,'Liquor_models') # 'D:/lys/studystudy/phd/0-Project_absorption_correction/dataset/Blender'
    if os.path.exists(loop_pth) is False:
        os.makedirs(loop_pth)
    if os.path.exists(crystal_pth) is False:
        os.makedirs(crystal_pth)
    if os.path.exists(liquor_pth) is False:
        os.makedirs(liquor_pth)

    csv_filename_overall=os.path.join(config['setting']['csv_save_dir'], 'overall.csv')
    csv_filename_crystal=config['converter']['csv_filename_crystal']
    csv_filename_loop=os.path.join(config['setting']['csv_save_dir'], 'loop.csv')
    csv_filename_liquor=config['converter']['csv_filename_liquor']

    
    with open(csv_filename_overall, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metadata_template.keys())
        writer.writeheader()
    
   

    if config['loop']['create_loop']:
        create_loop(config,loop_pth,save_obj=True)
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
    else:
        shutil.copy2(config['loop']['simiulate_loop'],loop_pth)
    
    shutil.copy2("config.yaml",config['setting']['base_root'])
    crystal_column_names = ['num_vertices', 'height', 'bottom_width', 'middle_width', 
                'top_width_in_percent', 'top_height_in_percent', 'scale_factor']
    crystal_row =['crystal_id']+ crystal_column_names+['rotation_tuple']+ ['crystal_path']
    
    if config['crystal']['create_crystal']:

        crystal_writer,crystal_writer_file = create_csv_writer(csv_filename_crystal, crystal_row)
        crystal_dict = {key: None for key in crystal_row}
        create_crystal(config,loop_pth,crystal_pth,crystal_writer,crystal_writer_file,crystal_dict)
        crystal_writer_file.close()
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    if config['liquor']['create_liquor']:
        
        liquor_row = ['Data_id','overall_path','crystal_id']
        liquor_row += ['diffusion','surface_tension','frame','liquor_shape']

        crystal_csv= pd.read_csv(csv_filename_crystal)
        crystal_paths = crystal_csv['crystal_path'].tolist()
        crystal_id= crystal_csv['crystal_id'].tolist()
        
        liquor_writer,liquor_writer_file = create_csv_writer(csv_filename_liquor, liquor_row)
        liquor_dict = {key: None for key in liquor_row}
        
        create_liquor_fluid(config,loop_pth,crystal_paths,crystal_id,liquor_pth,liquor_writer,liquor_writer_file,liquor_dict)
        liquor_writer_file.close()
        