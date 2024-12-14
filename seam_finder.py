import numpy as np
import cupy as cp
from numba import cuda

@cuda.jit
def find_vertical_seam_kernel(energy_map, cumulative_map, backtrack):
    """CUDA kernel for finding vertical seam using dynamic programming."""
    x, y = cuda.grid(2)
    
    if x < energy_map.shape[1] and y < energy_map.shape[0]:
        if y == 0:
            cumulative_map[y, x] = energy_map[y, x]
        else:
            min_energy = cumulative_map[y-1, x]
            min_x = x
            
            if x > 0:
                left = cumulative_map[y-1, x-1]
                if left < min_energy:
                    min_energy = left
                    min_x = x-1
                    
            if x < energy_map.shape[1] - 1:
                right = cumulative_map[y-1, x+1]
                if right < min_energy:
                    min_energy = right
                    min_x = x+1
                    
            cumulative_map[y, x] = energy_map[y, x] + min_energy
            backtrack[y, x] = min_x

class SeamFinder:
    def __init__(self):
        self.device = cp.cuda.Device()
    
    def find_seam(self, energy_map):
        """Find optimal seam using GPU-accelerated dynamic programming."""
        height, width = energy_map.shape
        
        # Allocate memory on GPU
        energy_gpu = cp.asarray(energy_map)
        cumulative_map = cp.zeros_like(energy_gpu)
        backtrack = cp.zeros_like(energy_gpu, dtype=cp.int32)
        
        # Configure CUDA grid
        threadsperblock = (16, 16)
        blockspergrid_x = (width + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (height + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        
        # Launch kernel
        find_vertical_seam_kernel[blockspergrid, threadsperblock](
            energy_gpu, cumulative_map, backtrack
        )
        
        # Find minimum seam path
        seam = np.zeros(height, dtype=np.int32)
        seam[-1] = cp.argmin(cumulative_map[-1]).get()
        
        for i in range(height-2, -1, -1):
            seam[i] = backtrack[i+1, seam[i+1]].get()
            
        return seam