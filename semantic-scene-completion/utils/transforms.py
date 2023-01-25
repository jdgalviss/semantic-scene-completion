import torch

def get_bev(voxels):
    '''
    Receives a tensor of voxels of shape [1, W, L, H] and returns the Bird-eye-view tensor of size [1, W, L]
    '''
    voxels_copy = torch.clone(voxels)
    locs = torch.nonzero(voxels_copy)    
    locations_z = torch.zeros_like(voxels_copy).long()
    locations_z[locs[:,0],locs[:,1],locs[:,2],locs[:,3]] = locs[:,3] + 1
    max_locs_z = torch.max(locations_z,3)
    bev = torch.zeros_like(max_locs_z.values, dtype=voxels_copy.dtype)
    max_locs_nz = torch.nonzero(max_locs_z.values)
    bev[max_locs_nz[:,0], max_locs_nz[:,1], max_locs_nz[:,2]] = voxels_copy[max_locs_nz[:,0], max_locs_nz[:,1], max_locs_nz[:,2], max_locs_z.values[max_locs_nz[:,0], max_locs_nz[:,1], max_locs_nz[:,2]]-1]
    return bev