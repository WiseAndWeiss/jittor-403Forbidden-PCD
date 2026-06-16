import jittor as jt


def chamfer_distance(p1, p2):
    """
    Compute Chamfer distance between two point clouds.
    Args:
        p1: (B, N1, 3)
        p2: (B, N2, 3)
    Returns:
        dist1: (B, N1) squared distance from each point in p1 to nearest in p2
        dist2: (B, N2) squared distance from each point in p2 to nearest in p1
        idx1: (B, N1) indices of nearest neighbors in p2
        idx2: (B, N2) indices of nearest neighbors in p1
    """
    B, N1, _ = p1.shape
    _, N2, _ = p2.shape

    diff = p1.unsqueeze(2) - p2.unsqueeze(1)  # (B, N1, N2, 3)
    dist = (diff ** 2).sum(dim=-1)  # (B, N1, N2)

    dist1 = dist.min(dim=-1)   # (B, N1), nearest in p2 for each p1 point
    dist2 = dist.min(dim=-2)   # (B, N2), nearest in p1 for each p2 point
    idx1 = idx2 = None
    # Actually: dist.min(dim=-1) gives min over p2 (N2) → (B, N1)
    # dist.min(dim=-2) gives min over p1 (N1) → (B, N2). But wait, let me re-check.
    # dist is (B, N1, N2). dim=-1 is N2, dim=-2 is N1.
    # dist.min(dim=-1) → min over p2 → (B, N1). This is correct: for each p1 point, nearest p2.
    # dist.min(dim=-2) → min over p1 → (B, N2). But we need: for each p2 point, nearest p1.
    # dist[:, :, i] gives distances from p1[:,:,None] to p2[:,i,:] → shape (B, N1) for each p2 point.
    # So min over N1 (dim=1) gives the nearest p1 for each p2.
    # In the unsqueezed form: diff (B, N1, N2, 3), dist (B, N1, N2)
    # dist.min(dim=-1) → (B, N1) - for each p1, nearest p2. ✓
    # But for p2→p1, we need: for each p2, nearest p1. That's min over N1 (dim=1, not dim=-2 which is the same).
    # dist.min(dim=1) → (B, N2). Wait no, dim=-2 is the same as dim=1 for a 3D tensor.
    # dist is (B, N1, N2). dim=-2 is N1, dim=-1 is N2.
    # min over dim=1 (N1) gives (B, N2). But dist[b, :, j] is distances from all p1 to p2_j.
    # So min over N1 gives nearest p1 for each p2_j. This is what we want for dist2.

    return dist1, dist2, idx1, idx2


def calc_cd_like_InfoV2(p1, p2):
    """
    InfoCD loss, matching the official implementation exactly.

    Args:
        p1: (B, N, 3) predicted points
        p2: (B, M, 3) clean points
    Returns:
        scalar loss
    """
    dist1, dist2, idx1, idx2 = chamfer_distance(p1, p2)

    dist1 = dist1.maximum(1e-9)
    dist2 = dist2.maximum(1e-9)

    d1 = jt.sqrt(dist1)  # (B, N)
    d2 = jt.sqrt(dist2)  # (B, M)

    # distances1 = -log(exp(-0.5*d1) / (sum(exp(-0.5*d1)))**1e-7)
    sum_exp1 = jt.exp(-0.5 * d1).sum(dim=-1, keepdims=True) + 1e-7  # (B, 1)
    distances1 = -jt.log(
        jt.exp(-0.5 * d1) / (sum_exp1 ** 1e-7) + 1e-7
    )

    sum_exp2 = jt.exp(-0.5 * d2).sum(dim=-1, keepdims=True) + 1e-7  # (B, 1)
    distances2 = -jt.log(
        jt.exp(-0.5 * d2) / (sum_exp2 ** 1e-7) + 1e-7
    )

    return (distances1.sum() + distances2.sum()) / (2 * p1.shape[0])
