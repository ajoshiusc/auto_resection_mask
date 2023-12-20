import numpy as np
from scipy.spatial import ConvexHull
from noise import pnoise3  # You may need to install the `noise` library using: pip install noise
import random


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes


def mesh_to_mask_3d(vertices, volume_size=(256, 256, 256)):
    # Create an empty 3D binary mask
    mask_3d = np.zeros(volume_size, dtype=np.uint8)

    # Create a list of polygons from the mesh triangles
    #polygons = [vertices[triangle] for triangle in triangles]

    # Convert the 3D mesh to voxel coordinates
    voxel_coords = np.round(vertices).astype(int)
    voxel_coords[:,0] = np.clip(voxel_coords[:,0], 1, volume_size[0] - 2)
    voxel_coords[:,1] = np.clip(voxel_coords[:,1], 1, volume_size[1] - 2)
    voxel_coords[:,2] = np.clip(voxel_coords[:,2], 1, volume_size[2] - 2)

    mask_3d[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = 1

    # Draw filled polygons in 3D mask
    #for poly in polygons:
    #    rr, cc, dd = polygon(voxel_coords[:, 1], voxel_coords[:, 0], voxel_coords[:, 2], shape=volume_size)
    #    mask_3d[rr, cc, dd] = 1

    # Use binary_fill_holes to fill any holes in the 3D mask
    filled_mask_3d = binary_fill_holes(mask_3d)

    return filled_mask_3d







# Function to generate Perlin noise
def generate_perlin_noise(x, y, z):
    return pnoise3(x, y, z)

def generate_icosphere_mesh(radius, num_subdivisions):
    # Create icosphere vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0

    vertices = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ])

    # Normalize vertices to lie on the sphere
    vertices /= np.linalg.norm(vertices, axis=1)[:, np.newaxis]

    # Create icosphere faces
    #faces = ConvexHull(vertices).simplices

    faces = np.array([
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1]
    ])

    # Subdivide the icosphere
    for _ in range(num_subdivisions):
        new_faces = []
        for face in faces:
            v1, v2, v3 = face
            v12 = (vertices[v1] + vertices[v2]) / 2.0
            v23 = (vertices[v2] + vertices[v3]) / 2.0
            v31 = (vertices[v3] + vertices[v1]) / 2.0

            v12 /= np.linalg.norm(v12)
            v23 /= np.linalg.norm(v23)
            v31 /= np.linalg.norm(v31)

            vertices = np.vstack([vertices, v12, v23, v31])
            new_faces.extend([
                [v1, len(vertices) - 3, len(vertices) - 1],
                [v2, len(vertices) - 2, len(vertices) - 3],
                [v3, len(vertices) - 1, len(vertices) - 2],
                [len(vertices) - 3, len(vertices) - 2, len(vertices) - 1]
            ])

        faces = np.array(new_faces)

    return vertices, faces



def gen_rand_blob_mesh(radius=1.0,num_subdivisions=3,scale=0.3):
    # Generate the icosphere mesh
    vertices, faces = generate_icosphere_mesh(radius, num_subdivisions)

    # Add Perlin noise to vertices
    a = random.random()*10
    for i in range(vertices.shape[0]):
        noise_val = generate_perlin_noise(vertices[i, 0]+a, vertices[i, 1], vertices[i, 2])
        vertices[i, 0] *= 1+ scale*noise_val 
        vertices[i, 1] *= 1+ scale*noise_val
        vertices[i, 2] *= 1+ scale*noise_val

    return vertices, faces