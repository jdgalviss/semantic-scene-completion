import torch

def get_bev(voxels):
    '''
    Receives a tensor of voxels of shape [1, W, L, H] and returns the Bird-eye-view tensor of size [1, W, L]
    '''
    voxels_copy = torch.clone(voxels)
    locs = torch.nonzero(voxels_copy)    
    locations_z = torch.zeros_like(voxels_copy).long()
    locations_z[locs[:,0],locs[:,1],locs[:,2],locs[:,3]] = locs[:,3] + 1
    # locations_z[voxels_copy == 255] = -1
    max_locs_z = torch.max(locations_z,3)
    bev = torch.zeros_like(max_locs_z.values, dtype=voxels_copy.dtype)
    max_locs_nz = torch.nonzero(max_locs_z.values)
    bev[max_locs_nz[:,0], max_locs_nz[:,1], max_locs_nz[:,2]] = voxels_copy[max_locs_nz[:,0], max_locs_nz[:,1], max_locs_nz[:,2], max_locs_z.values[max_locs_nz[:,0], max_locs_nz[:,1], max_locs_nz[:,2]]-1]
    return bev

def get_2d_input(features, locs):
    '''
    Receives a tensor of features of shape [1, W, L, H] as well as a 
     locations tensor with the voxels corresponding to points from the pointcloud
    
    Returns the tensor of features of size [1, W, L]
    '''

    locations_z = torch.ones_like(features)*-1.0
    locations_z[:,locs[:,0],locs[:,1],locs[:,2]] = locs[:,2].float()
    max_locs_voxels = torch.max(locations_z.float(),3)[0]
    max_locs_voxels[max_locs_voxels==-1] = 255
    max_locs_voxels[max_locs_voxels!=255] = max_locs_voxels[max_locs_voxels!=255]/32.0 - 0.5

    locations_z = torch.ones_like(features)*255.0
    locations_z[:,locs[:,0],locs[:,1],locs[:,2]] = locs[:,2].float()
    min_locs_voxels = torch.min(locations_z.float(),3)[0]
    min_locs_voxels[min_locs_voxels!=255]  = min_locs_voxels[min_locs_voxels!=255]/32.0 -0.5

    locations_z = torch.full(features.shape, torch.nan)
    locations_z[:,locs[:,0],locs[:,1],locs[:,2]] = locs[:,2].float()
    mean_locs_voxels = torch.nanmean(locations_z.float(),3)
    mean_locs_voxels = mean_locs_voxels/6.0 - 0.5
    mean_locs_voxels = torch.nan_to_num(mean_locs_voxels,255)

    feature_voxels_cpy = torch.ones_like(features)*-255
    feature_voxels_cpy[:,locs[:,0],locs[:,1],locs[:,2]] = features[:,locs[:,0],locs[:,1],locs[:,2]]
    max_feature_voxels = torch.max(features,3)[0].float()
    max_feature_voxels[max_feature_voxels==-255] = 255
    # max_feature_voxels[max_feature_voxels!=255] = max_feature_voxels[max_feature_voxels!=255]/2.0 - 0.5

    feature_voxels_cpy = torch.ones_like(features)*255
    feature_voxels_cpy[:,locs[:,0],locs[:,1],locs[:,2]] = features[:,locs[:,0],locs[:,1],locs[:,2]]
    min_feature_voxels = torch.min(feature_voxels_cpy,3)[0].float()
    # min_feature_voxels[min_feature_voxels != 255] = min_feature_voxels[min_feature_voxels != 255]/1.0 - 0.5

    feature_voxels_cpy = torch.full(features.shape, torch.nan)
    feature_voxels_cpy[:,locs[:,0],locs[:,1],locs[:,2]] = features[:,locs[:,0],locs[:,1],locs[:,2]]
    mean_feature_voxels = torch.nanmean(features,3).float()
    mean_feature_voxels = mean_feature_voxels/0.7 - 0.5
    mean_feature_voxels = torch.nan_to_num(mean_feature_voxels,255)

    density_feature_voxels = torch.sum(torch.logical_not(torch.isnan(feature_voxels_cpy)),3).float()/16.0 - 0.5

    features2d = torch.cat([min_locs_voxels,max_locs_voxels,mean_locs_voxels,min_feature_voxels,
                            max_feature_voxels,mean_feature_voxels, density_feature_voxels])
    
    return features2d