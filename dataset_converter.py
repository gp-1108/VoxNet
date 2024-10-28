import numpy as np
import trimesh
from scipy.ndimage import zoom
from scipy.ndimage import binary_fill_holes
import zstandard as zstd
import pickle
import os
from multiprocessing import Pool, cpu_count, Queue, current_process, TimeoutError
import argparse
import logging
import logging.handlers
import threading
import time

class QueueLogger:
    """
    Logger class to handle multiprocessing logging using a queue.
    """
    def __init__(self, log_file_path):
        self.queue = Queue()
        self.log_file_path = log_file_path
        self.listener_thread = threading.Thread(target=self._listener, daemon=True)
        self.listener_thread.start()

    def _listener(self):
        with open(self.log_file_path, 'a') as f:
            while True:
                record = self.queue.get()
                if record is None:
                    break
                f.write(record + '\n')
                f.flush()

    def log(self, level, message):
        pid = current_process().pid
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        record = f"{timestamp} PID {pid} [{level}]: {message}"
        self.queue.put(record)

    def close(self):
        self.queue.put(None)
        self.listener_thread.join()

logger = None

def load_off(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Ensure it's an OFF file
    assert lines[0].strip() == 'OFF', "The file is not an OFF file."

    # Get number of vertices and faces
    parts = lines[1].strip().split()
    num_vertices = int(parts[0])
    num_faces = int(parts[1])

    # Read vertices
    vertices = []
    for i in range(2, 2 + num_vertices):
        vertex = list(map(float, lines[i].strip().split()))
        vertices.append(vertex)

    # Read faces
    faces = []
    for i in range(2 + num_vertices, 2 + num_vertices + num_faces):
        face = list(map(int, lines[i].strip().split()[1:]))  # Skip the first number (face size)
        faces.append(face)

    vertices = np.array(vertices)
    faces = np.array(faces)
    
    return vertices, faces

def voxelize_fill_volume(vertices, faces, resolution=32):
    # Get the bounding box of the mesh
    min_bounds = np.min(vertices, axis=0)
    max_bounds = np.max(vertices, axis=0)
    bounds_size = max_bounds - min_bounds

    # Calculate scaling factor to fit the object inside the voxel grid without distorting proportions
    max_extent = np.max(bounds_size)
    scale_factors = resolution / max_extent

    # Create an empty voxel grid
    voxel_grid = np.zeros((resolution, resolution, resolution), dtype=bool)

    # Loop over faces and mark surface voxels
    for face in faces:
        triangle = vertices[face]  # Get the vertices of the triangle

        # Scale the vertices according to their original proportions
        scaled_triangle = (triangle - min_bounds) * scale_factors

        # Find the min and max coordinates of the scaled triangle
        tri_min_vox = np.floor(np.min(scaled_triangle, axis=0)).astype(int)
        tri_max_vox = np.ceil(np.max(scaled_triangle, axis=0)).astype(int)

        # Clamp the values to the grid size
        tri_min_vox = np.clip(tri_min_vox, 0, resolution - 1)
        tri_max_vox = np.clip(tri_max_vox, 0, resolution - 1)

        # Fill the voxel grid for each triangle within its bounding box (surface only)
        voxel_grid[tri_min_vox[0]:tri_max_vox[0]+1, 
                   tri_min_vox[1]:tri_max_vox[1]+1, 
                   tri_min_vox[2]:tri_max_vox[2]+1] = True
    return voxel_grid

def fill_volume(voxel_grid):
    # Fill the volume by treating the surface as boundaries
    filled_grid = binary_fill_holes(voxel_grid).astype(bool)
    return filled_grid


def fit_voxel_grid(voxel_grid, n):
    # Get the current resolution of the input voxel grid
    current_resolution = max(voxel_grid.shape)

    # Create an empty voxel grid with new resolution (n x n x n)
    new_voxel_grid = np.zeros((n, n, n), dtype=bool)

    # Calculate the scaling factor to fit the current grid into the new grid
    scale_factor = n / current_resolution

    # Loop over each filled voxel in the original grid and map it to the new grid
    filled_voxels = np.argwhere(voxel_grid)
    for voxel in filled_voxels:
        new_voxel = np.floor(voxel * scale_factor).astype(int)
        new_voxel = np.clip(new_voxel, 0, n - 1)
        new_voxel_grid[new_voxel[0], new_voxel[1], new_voxel[2]] = True

    return new_voxel_grid


def set_axes_equal(ax):
    """Set equal scaling for 3D axes."""
    extents = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    centers = np.mean(extents, axis=1)
    max_range = np.ptp(extents, axis=1).max() / 2

    for ctr, axis in zip(centers, 'xyz'):
        getattr(ax, f'set_{axis}lim')(ctr - max_range, ctr + max_range)


def from_off_to_voxel(off_file_path, resolution=64, target_shape=32, compression_level=22, zst_output_path=None):
    # Load and voxelize the mesh
    vertices, faces = load_off(off_file_path)
    surface_voxel_grid = voxelize_fill_volume(vertices, faces, resolution=resolution)

    # Fill the volume
    filled_voxel_grid = fill_volume(surface_voxel_grid)
    filled_voxel_grid = fit_voxel_grid(filled_voxel_grid, n=target_shape)
    # Serialize and compress the voxel grid
    cctx = zstd.ZstdCompressor(level=compression_level)
    os.makedirs(os.path.dirname(zst_output_path), exist_ok=True)  # Ensure output directory exists
    with open(zst_output_path, 'wb') as f:
        f.write(cctx.compress(pickle.dumps(filled_voxel_grid)))


def load_paths(folder_path, output_path):
    """
    Load all paths for .off files in the ModelNet dataset matching the given folder path
    in the output path.
    """
    input_paths = []
    output_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".off"):
                input_paths.append(os.path.join(root, file))
                output_paths.append(os.path.join(output_path, root[len(folder_path)+1:], file[:-4] + ".zst"))
    return input_paths, output_paths

def process_file(paths, resolution, target_shape, compression_level):
    """ Helper function to compress an .off file to .zst. """
    off_file_path, zst_output_path = paths
    return from_off_to_voxel(off_file_path, resolution, target_shape, compression_level, zst_output_path)

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Compress .off files to .zst format with voxelization.")
    
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the input directory containing .off files.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory where .zst files will be saved.")
    parser.add_argument('--resolution', type=int, default=64, help="The resolution for voxelization. Default is 64.")
    parser.add_argument('--target_size', type=int, default=32, help="The desired shape for the voxel grid after rescaling. Default is 32")
    parser.add_argument('--compression_level', type=int, default=22, help="The compression level for Zstandard. Default is 22.")
    parser.add_argument('--num_workers', type=int, default=cpu_count(), help="Number of parallel processes. Default is the number of CPU cores.")

    return parser.parse_args()

def process_with_timeout(pool, paths, resolution, target_shape, compression_level, timeout=300):
    """
    Process files with a timeout.

    Parameters:
    - pool (multiprocessing.Pool): The multiprocessing pool.
    - paths (list of tuples): List of paths to process.
    - timeout (int): Timeout in seconds for each process. Default is 300 seconds (5 minutes).
    """
    results = []
    for path in paths:
        result = pool.apply_async(process_file, (path, resolution, target_shape, compression_level))
        try:
            logger.log('INFO', f"Processing {path[0]}")
            results.append(result.get(timeout=timeout))
            logger.log('INFO', f"Processing {path[0]}")
        except TimeoutError:
            logger.log('ERROR', f"Processing timed out for: {path[0]}")
            results.append(path[0])  # Mark the file as failed
    return results

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()

    # Set up logging
    logger = QueueLogger("process.log")

    # Load all paths for .off files in the input directory
    input_paths, output_paths = load_paths(args.input_dir, args.output_dir)

    # Create a list of tuples, each containing the input and output path
    paths = list(zip(input_paths, output_paths))

    # Set up a multiprocessing Pool
    with Pool(processes=args.num_workers) as pool:
        # Process files with timeout handling
        failed_files = process_with_timeout(pool, paths, args.resolution, args.target_size, args.compression_level)

    # Filter out None values (successful files) from failed_files list
    failed_files = [f for f in failed_files if f]

    if failed_files:
        logger.log('ERROR', f"{len(failed_files)} files failed to process. Failed files: {failed_files}")
    else:
        logger.log('INFO', "All files processed successfully.")

    # Close the logger
    logger.close()
