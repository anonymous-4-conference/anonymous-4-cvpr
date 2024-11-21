import open3d as o3d
import numpy as np
import os 
import yaml
import pandas as pd
import pdb
from collada import Collada
import h5py
import matplotlib.pyplot as plt
import cv2
import skimage.io as io 
global color_map    
import time 

color_map = {
        0: [0, 0, 0],       # Dark
        1: [255, 0, 0],     # Blue
        2: [0, 255, 255],   # Yellow
        3: [0, 255, 0]      # Green
    }
def mask_filled_optimized(image, value=None):
    """
    Fill the frame with pixel values.
    """
    # Find non-zero pixels (border points)
    lzx, lzy = np.nonzero(image)

    # Create a mask image initialized to zero
    mask = np.zeros_like(image)

    # Iterate over unique y-coordinates
    for y in np.unique(lzy):
        # Find the min and max x-coordinates for this y
        x_coords = lzx[lzy == y]
        x_min, x_max = np.min(x_coords), np.max(x_coords)

        # Fill between min and max x-coordinates
        mask[x_min:x_max+1, y] = value

    # Combine original image and mask
    filled_image = np.maximum(image, mask)

    return filled_image

def mask_filled(image,value=None):
    """
    as the svg files have the shape/frame of the object,
    but not the filled shape, so fill the frame with pixel values
    """
    lzx, lzy = np.nonzero(image)
    lz = list(zip(lzx, lzy))
    y_past = 0
    for x, y in lz:
        # to save some time to ignore repeated x coordinates
        if y == y_past:
            continue

        yes = []
        for i in lz:
            if i[1] == y:
                yes.append(i[0])
        image[yes[0]:yes[-1], y] = value
        y_past = y
    return image


def fill_mask(img,value,_cls='li'):
        
    mask = np.uint8(img == value)
    output = np.zeros_like(img)
 

    if _cls == 'cr':
        kernel_cr = np.ones((10,10), np.uint8)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_cr)
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, value, thickness=cv2.FILLED)
        return output
    if _cls == 'lo':
        # pdb.set_trace()
        kernel_lo = np.ones((10,10), np.uint8)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_lo)
        kernel_lo_vertical = np.ones((100,1), np.uint8)
        closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel_lo_vertical)
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, value, thickness=cv2.FILLED)
        
        return output
    if _cls == 'li':                
        # _, binary = cv2.threshold(closed, 1, 255, cv2.THRESH_BINARY)
        # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kernel_li = np.ones((10,10), np.uint8)
        
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_li)
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(output, contours, -1, 255, thickness=-1)
        cv2.drawContours(output, contours, -1, color=value, thickness=cv2.FILLED)
        
        return output
        filled_image = np.zeros((img.shape[0],closing.shape[1]), dtype=np.uint8)
        
        # for contour in contours:
        #     cv2.fillPoly(filled_image, pts=[contour], color=(255))
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if len(contours) == 0:
            cv2.drawContours(output, contours, -1, value, thickness=cv2.FILLED)
            return output
        largest_contour = contours[0]
        y_value = largest_contour[len(largest_contour)//2][0][0]
        target =  largest_contour[largest_contour[:, 0, 0] == y_value]
        min_x = np.min(target[:, 0, 1])
        max_x = np.max(target[:, 0, 1])
        x = (min_x + max_x) // 2
        center = (int(y_value),int(x) )


        # Prepare the mask for floodFill
        h, w = img.shape[:2]
        mask_fluid = np.zeros((h+2, w+2), np.uint8)

        filled_image = img.copy()
        cv2.floodFill(filled_image, mask_fluid, center, (255,0,0))
        filled_image[filled_image != 255] = 0
        filled_image[filled_image == 255] = value
        pdb.set_trace()
        # filled_image =mask_filled_optimized(closed,value=value)
        return filled_image
    
def save_slice_to_png(voxel_array, save_dir='./test_2'):
    li_grid,lo_grid,cr_grid = voxel_array

    new_array_shape = (li_grid.shape[0], li_grid.shape[1] + 100, li_grid.shape[2] + 100)
    new_array = np.zeros(new_array_shape, dtype=np.uint8)
    padding = ((50, 50), (50, 50))
    # pdb.set_trace()
    print('save_dir:',save_dir)
    print('li_grid.shape:',li_grid.shape)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in range(li_grid.shape[0]):
        # i =200
        
        li_slice = np.pad(li_grid[i,:,:], pad_width=padding, mode='constant', constant_values=0)
        lo_slice = np.pad(lo_grid[i,:,:], pad_width=padding, mode='constant', constant_values=0)
        cr_slice = np.pad(cr_grid[i,:,:], pad_width=padding, mode='constant', constant_values=0)
        sum_slice = li_slice+lo_slice+cr_slice
        if np.max(sum_slice) == 0:
            # print('slice_z{} is empty'.format(i))
            # pdb.set_trace()
            continue

        # if i > 
        rgb_img = np.zeros((li_slice.shape[0], li_slice.shape[1], 3), dtype=np.uint8)
        li_img = np.zeros((li_slice.shape[0], li_slice.shape[1], 3), dtype=np.uint8)
        liquor_mask =fill_mask(li_slice,1,_cls='li')
        loop_mask =fill_mask(lo_slice,2,_cls='lo')
        crystal_mask =fill_mask(cr_slice,3,_cls='cr')
        overlap_mask =np.max([liquor_mask,loop_mask,crystal_mask],axis=0)
        new_array[i,:,:]=overlap_mask
        # img = voxel_array[171,:,:]
        # mask = np.uint8(img == 2)
        # output = np.zeros_like(img)
        # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(output, contours, -1, 2, thickness=cv2.FILLED)
        # rgb_img[output == 2] = [255, 255, 0]
        # cv2.imwrite('./filled_image.png', rgb_img)
        # Map each voxel value to its corresponding color
        cv2.imwrite(os.path.join(save_dir, 'slice_uint_{}.tiff'.format(i)), overlap_mask)
        for value, color in color_map.items():
            rgb_img[overlap_mask == value] = color

        
        cv2.imwrite(os.path.join(save_dir, 'slice_rgb_{}.png'.format(i)), rgb_img)
        # for value, color in color_map.items():
        #     li_img[liquor_mask == value] = color
        # cv2.imwrite(os.path.join(save_dir, 'slice_z{}_li.png'.format(i)), li_img)
        # print('save slice_z{}.png'.format(i))
    return new_array
def is_mesh_inside_other(mesh1, mesh2):
    """
    Check if any vertex of mesh1 is inside mesh2.
    """
    for vertex in np.asarray(mesh1.vertices):
        if mesh2.is_point_inside(vertex):
            return True
    return False

def meshes_intersect(mesh1, mesh2):
    """
    Check if two meshes intersect.
    """
    return is_mesh_inside_other(mesh1, mesh2) or is_mesh_inside_other(mesh2, mesh1)


def merge_meshes(mesh1, mesh2):
    combined_mesh = o3d.geometry.TriangleMesh()
    if meshes_intersect(mesh1, mesh2):
        return None
    else:
        combined_mesh = mesh1 + mesh2
        return combined_mesh

def save_visualize_mesh(mesh,filename='mesh.png'):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # Create an offscreen window
    vis.add_geometry(mesh)

    # Render the scene and capture the image
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(False)

    # Save the captured image
    o3d.io.write_image(filename, image)

    # Destroy the visualizer window
    vis.destroy_window()

def visualize_mesh(open3d_mesh):
    vis = o3d.visualization.Visualizer()
    vis.create_window('Open3D Mesh Visualization', width=800, height=600)
    for mesh in open3d_mesh:
        vis.add_geometry(mesh)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0,0,0])
    vis.add_geometry(coord_frame)
    vis.run()  
    vis.destroy_window()

def load_dae_to_open3d_mesh(file_path):
    collada_mesh = Collada(file_path)
    geometry = collada_mesh.geometries[0]
    triangles = geometry.primitives[0]
    
    vertices = np.array(triangles.vertex)
    faces = np.array(triangles.vertex_index)
    
    open3d_mesh = o3d.geometry.TriangleMesh()
    open3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    open3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    
    open3d_mesh.compute_vertex_normals()
   
    return open3d_mesh



def mesh_to_voxel(mesh, voxel_size=1):
    """
    Convert a mesh to a voxel grid.
    :param mesh: The input mesh.
    :param voxel_size: The size of the voxel (smaller for higher resolution).
    :return: VoxelGrid object.
    """
    # # if the mesh is at low quality 
    # point_cloud = mesh.sample_points_poisson_disk(number_of_points=500)
    
    # # 
    # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size)

    
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
                                                              voxel_size=voxel_size)
    # pdb.set_trace()
    # voxel_filename = "test_voxel_grid.ply"
    # o3d.io.write_voxel_grid(voxel_filename, voxel_grid)
    # o3d.io.write_voxel_grid('liquor.ply', liquor_voxel_grid)
    # o3d.io.write_voxel_grid('loop.ply', loop_voxel_grid)
    # o3d.io.write_voxel_grid('crystal.ply', crystal_voxel_grid)
    return voxel_grid



def paint_voxels(voxel_grid, color):
    for voxel in voxel_grid.get_voxels():
        voxel.color = np.array(color)
   
    return voxel_grid
def create_combined_voxel_grid(liquor_dae, loop_sample, crystal_dae, voxel_size):
    # Compute dimensions for the combined grid
 
    # max_bound = np.maximum(np.maximum(liquor_voxel.get_max_bound(), loop_voxel.get_max_bound()), crystal_voxel.get_max_bound())
    # min_bound = np.minimum(np.minimum(liquor_voxel.get_min_bound(), loop_voxel.get_min_bound()), crystal_voxel.get_min_bound())
    # dims = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
    # loop_max_bound = loop_voxel.get_max_bound()
    # loop_min_bound = loop_voxel.get_min_bound()
    # crystal_max_bound = crystal_voxel.get_max_bound()
    # crystal_min_bound = cry   stal_voxel.get_min_bound()
    # max_bound = np.maximum(loop_max_bound, crystal_max_bound)
    # min_bound = np.minimum(loop_min_bound, crystal_min_bound)
    # dims = np.ceil((max_bound - min_bound) / voxel_size*1.2).astype(int)
    # Create an empty numpy array for the combined grid
    
    loop_mesh = load_dae_to_open3d_mesh(loop_sample)
    crystal_mesh = load_dae_to_open3d_mesh(crystal_dae)
    
    liquor_mesh = load_dae_to_open3d_mesh(liquor_dae)
    

    rotation_angle_radians = np.radians(90)
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, rotation_angle_radians, 0])
    # liquor_mesh.rotate(rotation_matrix, center=liquor_mesh.get_center())
    # crystal_mesh.rotate(rotation_matrix, center=crystal_mesh.get_center())
    liquor_mesh.rotate(rotation_matrix, center=liquor_mesh.get_center())
    
    # rotation_angle_radians = np.radians(-90)
    # rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, rotation_angle_radians, 0])
    # crystal_mesh.rotate(rotation_matrix, center=np.zeros(3))

    loop_max_bound_mesh=loop_mesh.get_axis_aligned_bounding_box().max_bound
    loop_min_bound_mesh=loop_mesh.get_axis_aligned_bounding_box().min_bound
    loop_middle_mesh=(loop_max_bound_mesh-loop_min_bound_mesh)/2
    
    crystal_max_bound_mesh=crystal_mesh.get_axis_aligned_bounding_box().max_bound
    crystal_min_bound_mesh=crystal_mesh.get_axis_aligned_bounding_box().min_bound
    crystal_middle_mesh=(crystal_max_bound_mesh-crystal_min_bound_mesh)/2
    
    liquor_max_bound_mesh=liquor_mesh.get_axis_aligned_bounding_box().max_bound
    liquor_min_bound_mesh=liquor_mesh.get_axis_aligned_bounding_box().min_bound
    liquor_middle_mesh=(liquor_max_bound_mesh-liquor_min_bound_mesh)/2
    
    loop_voxel = mesh_to_voxel(loop_mesh, voxel_size=1)
    crystal_voxel = mesh_to_voxel(crystal_mesh, voxel_size=1)
    liquor_voxel = mesh_to_voxel(liquor_mesh, voxel_size=1)
    
    loop_voxel_list=np.array([voxel.grid_index for voxel in loop_voxel.get_voxels()])
    crystal_voxel_list=np.array([voxel.grid_index for voxel in crystal_voxel.get_voxels()])
    liquor_voxel_list=np.array([voxel.grid_index for voxel in liquor_voxel.get_voxels()])
    
    # loop_max_bound = np.max(loop_voxel_list,axis=0)
    # loop_min_bound = np.min(loop_voxel_list,axis=0)
    # loop_middle=(loop_max_bound+loop_min_bound)/2

    # crystal_max_bound = np.max(crystal_voxel_list,axis=0)
    # crystal_min_bound = np.min(crystal_voxel_list,axis=0)
    # crystal_middle=(crystal_max_bound+crystal_min_bound)/2

    # liquor_max_bound = np.max(liquor_voxel_list,axis=0)
    # liquor_min_bound = np.min(liquor_voxel_list,axis=0)
    # liquor_middle=(liquor_max_bound+liquor_min_bound)/2
    # print(f'loop max bound:{loop_max_bound},crystal max bound:{crystal_max_bound},liquor max bound:{liquor_max_bound}')
    # print(f'loop min bound:{loop_min_bound},crystal min bound:{crystal_min_bound},liquor min bound:{liquor_min_bound}')
    # print(f'loop middle:{loop_middle},crystal middle:{crystal_middle},liquor middle:{liquor_middle}')
    # shift_base_lo_cr = np.maximum(loop_middle,crystal_middle)
    # shift_base = np.maximum(shift_base_lo_cr,liquor_middle) 
    

    # print(f'loop max bound_mesh:{loop_max_bound_mesh},crystal max bound_mesh:{crystal_max_bound_mesh},liquor max bound_mesh:{liquor_max_bound_mesh}')
    # print(f'loop min bound_mesh:{loop_min_bound_mesh},crystal min bound_mesh:{crystal_min_bound_mesh},liquor min bound_mesh:{liquor_min_bound_mesh}')
    # print(f'loop middle_mesh:{loop_middle_mesh},crystal middle_mesh:{crystal_middle_mesh},liquor middle_mesh:{liquor_middle_mesh}')
    
    # print(f'loop max bound:{loop_max_bound},crystal max bound:{crystal_max_bound},liquor max bound:{liquor_max_bound}')

    loop_voxel_list = loop_voxel_list + loop_min_bound_mesh
    crystal_voxel_list = crystal_voxel_list + crystal_min_bound_mesh
    liquor_voxel_list = liquor_voxel_list + liquor_min_bound_mesh
    min_boudary = np.minimum(np.minimum(loop_min_bound_mesh,crystal_min_bound_mesh),liquor_min_bound_mesh)
    max_boundary = np.maximum(np.maximum(loop_max_bound_mesh,crystal_max_bound_mesh),liquor_max_bound_mesh)
    loop_voxel_list = loop_voxel_list - min_boudary
    crystal_voxel_list = crystal_voxel_list - min_boudary
    liquor_voxel_list = liquor_voxel_list - min_boudary
    # pdb.set_trace()
    # shift_base_lo_cr = np.maximum(loop_max_bound,crystal_max_bound)
    # shift_base = np.maximum(shift_base_lo_cr,liquor_max_bound) 

    # print(f'loop middle:{loop_middle},crystal middle:{crystal_middle},liquor middle:{liquor_middle},shift_base:{shift_base}')
    """#shift based on the max bound of the middle of loop or crystal
    # this is becuase open3d will shift the voxel grid to the origin baesd on their bound boxes
    """

    # max_bound_lo_cr = np.maximum(loop_max_bound, crystal_max_bound)
    # max_bound = np.maximum(max_bound_lo_cr, liquor_max_bound)
    # min_bound_lo_cr = np.minimum(loop_min_bound, crystal_min_bound)
    # min_bound = np.minimum(min_bound_lo_cr, liquor_min_bound)
    # min_bound = np.minimum(loop_min_bound, crystal_min_bound,liquor_min_bound)
    # dims = np.ceil((max_bound) / voxel_size*1.2).astype(int)
    dims = np.ceil((max_boundary-min_boudary ) / voxel_size*1.2).astype(int)


    # combined_grid = np.zeros((dims_z,dims_y,dims_x), dtype=np.uint8)
    cr_grid = np.zeros(dims, dtype=np.uint8)
    lo_grid = np.zeros(dims, dtype=np.uint8)
    li_grid = np.zeros(dims, dtype=np.uint8)
    # Function to fill the grid
    # pdb.set_trace()
    def fill_grid(voxel_list, _grid,value,shift_base=np.array([0,0,0])):
        for voxel in voxel_list:
            
            i, j, k = (voxel+shift_base).astype(int)
            # print(i,j,k)
            try:
                if _grid[i, j, k] < value:  # Only fill empty spaces
                    _grid[i, j, k] = value
            except:
                pdb.set_trace()
        return _grid
    # Fill the grid - order is important
    # if liquor_voxel is not None:
    # li_grid = fill_grid(liquor_voxel_list, li_grid,1,shift_base-liquor_middle)  # Liquor first
    # lo_grid = fill_grid(loop_voxel_list, lo_grid,2,shift_base-loop_middle)    # 
    # cr_grid = fill_grid(crystal_voxel_list, cr_grid,3,shift_base-crystal_middle) # 
    li_grid = fill_grid(liquor_voxel_list, li_grid,1)  # Liquor first
    lo_grid = fill_grid(loop_voxel_list, lo_grid,2)    # 
    cr_grid = fill_grid(crystal_voxel_list, cr_grid,3) # 

    # np.save('./li_grid.npy',li_grid)
    # np.save('./cr_grid.npy',cr_grid)
    # np.save('./lo_grid.npy',lo_grid)
    # pdb.set_trace()
    return li_grid,lo_grid,cr_grid
if __name__ == "__main__":
    global config
    with open('./config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    base_root = config['setting']['base_root']
    simulation_data_dir = os.path.join(config['setting']['simulation_data_dir'],'slices')
    loop_pth =os.path.join(base_root,'Loop_models') 
    crystal_pth =os.path.join(base_root,'Crystal_models')
    liquor_pth =os.path.join(base_root,'Liquor_models')
    
    if os.path.exists(simulation_data_dir) is False:
        os.makedirs(simulation_data_dir)
    # csv_filename_overall=os.path.join(config['setting']['csv_save_dir'], 'overall.csv')
    # csv_filename_crystal=os.path.join(config['setting']['csv_save_dir'], 'crystal.csv')
    # csv_filename_loop=os.path.join(config['setting']['csv_save_dir'], 'loop.csv')
    # csv_filename_liquor=os.path.join(config['setting']['csv_save_dir'], 'liquor_new.csv')
    csv_filename_crystal=config['converter']['csv_filename_crystal']
    csv_filename_liquor=config['converter']['csv_filename_liquor']

    crystal_csv = pd.read_csv(csv_filename_crystal)
    liquor_csv = pd.read_csv(csv_filename_liquor)

    crystal_paths = crystal_csv['crystal_path'].tolist()
    liquor_paths = liquor_csv['overall_path'].tolist()
    liquor_id = liquor_csv['crystal_id'].tolist()
    crystal_id= crystal_csv['crystal_id'].tolist()
    loop_sample=(os.path.join(loop_pth,config['loop']['simiulate_loop']))
    
    # crystal_dae=crystal_paths[480]
    # crystal_dae = '/home/yishun/projectcode/paper3/simulation/blender/dataset/Crystal_models/crystal_6_12_1_1_50_75.dae'
    # crystal_dae = '/home/yishun/projectcode/paper3/simulation/blender/dataset/Crystal_models/crystal_3_8_1_2_100_75.dae'
    # index_of_element = crystal_paths.index(crystal_dae)
    # liquor_dae=liquor_paths[index_of_element]
    # mesh = load_mesh_from_dae(loop_sample)
    counnter = 0
    t1 =time.time()
    # pdb.set_trace()
    for i, cr_id in enumerate(liquor_id):

        # 
        # crystal_dae = '/home/yishun/projectcode/paper3/simulation/blender/dataset/Crystal_models/crystal_3_8_1_2_100_75.dae'
        # p
        # li_index_of_element = liquor_id.index(crystal_id)
        cr_index_of_element =  crystal_id.index(cr_id)
        crystal_dae=crystal_paths[cr_index_of_element]
        liquor_dae=liquor_paths[i]
        
        save_name =  os.path.basename(crystal_dae).replace('.dae','').replace('crystal','slices')
        
        counnter +=1
        # if os.path.exists(os.path.join(simulation_data_dir,save_name)):
            # print('[{}]/[{}] {} is finished'.format(counnter,len(crystal_paths),save_name))
            # continue

        
        # liquor_dae = liquor_dae.replace('liquor_high_100_id000115.dae','liquor_crystal_3_8_1_2_100_75.dae')
        # crystal_dae = '/home/yishun/projectcode/paper3/simulation/blender/dataset/Crystal_models/crystal_3_8_1_2_100_75.dae'
        # pdb.set_trace()
        # loop_mesh = load_dae_to_open3d_mesh(loop_sample)
        # crystal_mesh = load_dae_to_open3d_mesh(crystal_dae)
        
        # liquor_mesh = load_dae_to_open3d_mesh(liquor_dae)
        # rotation_angle_radians = np.radians(-90)
       
        # rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, rotation_angle_radians, 0])
        # liquor_mesh.rotate(rotation_matrix, center=liquor_mesh.get_center())
      
        # loop_voxel_grid = mesh_to_voxel(loop_mesh, voxel_size=1)
        # crystal_voxel_grid = mesh_to_voxel(crystal_mesh, voxel_size=1)
        # liquor_voxel_grid = mesh_to_voxel(liquor_mesh, voxel_size=1)
        
        
        combined_voxel_grid = create_combined_voxel_grid(liquor_dae, loop_sample, crystal_dae, voxel_size=1)
        
        new_arr = save_slice_to_png(combined_voxel_grid,save_dir=os.path.join(simulation_data_dir,save_name))
        t2 = time.time()
       
        # counnter +=1
        # np.save(os.path.join(simulation_data_dir,'{}.npy'.format(save_name)),new_arr)
        print('[{}]/[{}] save {}, total time {}s '.format(counnter,len(crystal_paths),save_name,t2-t1))
        # pdb.set_trace()
        # pdb.set_trace()   

