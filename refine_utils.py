import torch
import torch.nn.functional as F
from snapy import MeshBlock

def refine_spatial(tensor: torch.Tensor, method: str):
    # Save original shape
    orig_shape = list(tensor.shape)
    n, m, p = orig_shape[-3], orig_shape[-2], orig_shape[-1]
    
    # Collapse all leading dims into a "batch" dim for F.interpolate
    # Shape becomes (-1, 1, n, m, p)
    flattened = tensor.reshape(-1, 1, n, m, p)
    
    # Upsample
    refined = F.interpolate(flattened, scale_factor=(2, 2, 1), mode=method)
    
    # Reshape back to (..., 2n, 2m, p)
    new_shape = orig_shape[:-3] + [2*n, 2*m, p]
    return refined.reshape(new_shape)

def coarsen_spatial(tensor: torch.Tensor):
    # Save original shape
    orig_shape = list(tensor.shape)
    n, m, p = orig_shape[-3], orig_shape[-2], orig_shape[-1]

    # Collapse all leading dims into a "batch" dim for F.interpolate
    # Shape becomes (-1, 1, n, m, p)
    flattened = tensor.reshape(-1, 1, n, m, p)
    
    # Downsample
    coarsened = F.interpolate(flattened, scale_factor=(0.5, 0.5, 1), mode='area')
    
    # Reshape back to (..., n/2, m/2, p)
    new_shape = orig_shape[:-3] + [n//2, m//2, p]
    return coarsened.reshape(new_shape)

def conservative_refine(x: torch.Tensor):
    y1 = refine_spatial(x, "trilinear")
    x1 = coarsen_spatial(y1)
    dy = refine_spatial(x - x1, "area")
    return y1 + dy

def conservative_coarsen(y: torch.Tensor):
    return coarsen_spatial(y)

def refine_meshblock(block: MeshBlock) -> MeshBlock:
    op = block.options()
    if op.coord().nx2() > 1:
        op.coord().nx2(op.coord().nx2() * 2)
    if op.coord().nx3() > 1:
        op.coord().nx3(op.coord().nx3() * 2)
    return MeshBlock(op)

def coarsen_meshblock(block: MeshBlock) -> MeshBlock:
    op = block.options()
    nghost = op.coord().nghost()
    if op.coord().nx2() > 1:
        op.coord().nx2(op.coord().nx2() // 2)
        assert op.coord().nx2() > nghost, "Cannot coarsen: insufficient cells in nx2"
    if op.coord().nx3() > 1:
        op.coord().nx3(op.coord().nx3() // 2)
        assert op.coord().nx3() > nghost, "Cannot coarsen: insufficient cells in nx3"
    return MeshBlock(op)

if __name__ == "__main__":
    # Test the function
    x = torch.randn(2, 3, 3, 2)  # Example tensor with shape (3, 4, 5, 6)
    for n in range(2):
        for k in range(3):
            for j in range(3):
                for i in range(2):
                    x[n,k,j,i] = n + k + j + i

    print("Original x :", x)
    y = conservative_refine(x)
    print("Refined y :", y)
    z = conservative_coarsen(y)
    print("Coarsened z :", z)

    assert torch.allclose(x, z), "The coarsened tensor does not match the original!"
