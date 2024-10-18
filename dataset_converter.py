import numpy as np
import trimesh
from scipy.ndimage import zoom
import zstandard as zstd
import pickle
import os
from multiprocessing import Pool, cpu_count
import argparse
import logging
from logger import Logger

def compress_off_to_zst(off_file_path, zst_output_path, pitch=0.6, target_shape=(32, 32, 32), compression_level=22):
    """
    Load a .off file, voxelize it, rescale it, and save it as a compressed .zst file.

    Parameters:
    - off_file_path (str): The path to the input .off file.
    - zst_output_path (str): The path to save the compressed .zst file.
    - pitch (float): The pitch (resolution) for voxelization. Default is 0.6.
    - target_shape (tuple of ints): The desired shape for the voxel grid after rescaling. Default is (32, 32, 32).
    - compression_level (int): The compression level for Zstandard. Default is 22.
    """

    try:
        # Load the .off file
        mesh = trimesh.load_mesh(off_file_path)

        # Voxelize the mesh
        voxelized = mesh.voxelized(pitch=pitch)
        solid_voxels = voxelized.fill()

        # Convert the voxel grid to a numpy array (3D)
        voxel_grid = solid_voxels.matrix

        # Calculate scaling factors
        scaling_factors = np.array(target_shape) / np.array(voxel_grid.shape)

        # Rescale the voxel grid to the desired shape
        rescaled_voxel_grid = zoom(voxel_grid, zoom=scaling_factors, order=0)

        # Convert rescaled voxel grid to binary (0 or 1)
        rescaled_voxel_grid = (rescaled_voxel_grid > 0).astype(np.int8)

        # Serialize and compress the voxel grid
        cctx = zstd.ZstdCompressor(level=compression_level)
        os.makedirs(os.path.dirname(zst_output_path), exist_ok=True)  # Ensure output directory exists
        with open(zst_output_path, 'wb') as f:
            f.write(cctx.compress(pickle.dumps(rescaled_voxel_grid)))

        logging.info(f"Successfully processed: {off_file_path}")

    except Exception as e:
        logging.error(f"Failed to process {off_file_path}: {e}")
        return off_file_path  # Return the path of the failed file

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

def process_file(paths):
    """ Helper function to compress an .off file to .zst. """
    off_file_path, zst_output_path = paths
    return compress_off_to_zst(off_file_path, zst_output_path)

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Compress .off files to .zst format with voxelization.")
    
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the input directory containing .off files.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory where .zst files will be saved.")
    parser.add_argument('--pitch', type=float, default=0.6, help="The pitch (resolution) for voxelization. Default is 0.6.")
    parser.add_argument('--target_shape', type=int, nargs=3, default=(32, 32, 32), help="The desired shape for the voxel grid after rescaling. Default is (32, 32, 32).")
    parser.add_argument('--compression_level', type=int, default=22, help="The compression level for Zstandard. Default is 22.")
    parser.add_argument('--num_workers', type=int, default=cpu_count(), help="Number of parallel processes. Default is the number of CPU cores.")

    return parser.parse_args()

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()

    # Load all paths for .off files in the input directory
    input_paths, output_paths = load_paths(args.input_dir, args.output_dir)

    # Create a list of tuples, each containing the input and output path
    paths = list(zip(input_paths, output_paths))

    # Set up a multiprocessing Pool
    with Pool(processes=args.num_workers) as pool:
        # Map the paths to the processing function
        failed_files = pool.map(process_file, paths)

    # Filter out None values (successful files) from failed_files list
    failed_files = [f for f in failed_files if f]

    if failed_files:
        logging.error(f"{len(failed_files)} files failed to process. Failed files: {failed_files}")
    else:
        logging.info("All files processed successfully.")
