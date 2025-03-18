import torch

def decimate(tensor, m):
    assert tensor.dim() == len(m)

    for d in range(tensor.dim()): 
        if m[d] is not None: 
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())
    return tensor

def save_checkpoint(epoch, model, optimizer):
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = 'my_pyramid_checkpoint.pth.tar'
    torch.save(state, filename)

def cxcy_to_xy(cxcy): # (8732, 4) # (cx, cy, w, h)
    return torch.cat([cxcy[:,:2] - (cxcy[:,2:] / 2), # x_min, y_min # cx - w/2, cy - h/2
                      cxcy[:,:2] + (cxcy[:,2:] / 2)], 1) # x_max, y_max # cx + w/2, cy + h/2

def find_intersection(set_1, set_2):
    # print(f"set_1.shape: {set_1.shape}, set_2.shape: {set_2.shape}")
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0) # (n1, n2)

    return intersection_dims[:,:,0] * intersection_dims[:,:,1] # (n1, n2)

def find_jaccard_overlap(set_1, set_2):
    intersection = find_intersection(set_1, set_2)

    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection # (n1, n2)

    return intersection / union # (n1, n2)