import math
import jittor as jt
from jittor import nn

from .blocks import block_decider


class FeatureExtraction(nn.Module):

    def __init__(self, d_in=0, d_out=32, n_cls=3, nsample=16,
                 stride_list=None, stride_dim_list=None, architecture=None):
        super().__init__()

        if architecture is None:
            architecture = ['startblock',
                            'downsample', 'downsample', 'downsample', 'downsample',
                            'upsample', 'upsample', 'upsample', 'upsample']
        if stride_list is None:
            stride_list = [4, 3, 2, 1]
        if stride_dim_list is None:
            stride_dim_list = [1.5, 1.5, 1.5, 1.5]

        self.d_in = d_in
        self.d_out = d_out
        self.n_cls = n_cls
        self.nsample = nsample
        self.stride_list = stride_list
        self.L = 4
        self.gamma = 0.396

        stride = 1
        stride_idx = 0
        d_prev = d_in

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []

        for block_name in architecture:
            if 'downsample' in block_name:
                self.encoder_skip_dims.append(d_prev)
                stride = stride_list[stride_idx]
                d_out = int(d_out * stride_dim_list[stride_idx])
                stride_idx += 1
                self.encoder_blocks.append(
                    block_decider(block_name)(d_prev, d_out, nsample, stride)
                )
            elif 'upsample' in block_name:
                skip_dim = self.encoder_skip_dims.pop()
                d_out = skip_dim
                self.decoder_blocks.append(
                    block_decider(block_name)([d_prev, skip_dim], d_out, nsample)
                )
            else:
                self.encoder_blocks.append(
                    block_decider(block_name)(d_prev, d_out, nsample, stride)
                )
            d_prev = d_out

        self.linear0_1 = nn.Linear(d_out, 128, bias=False)
        self.linear0_2 = nn.Linear(128, 64)
        self.linear0_3 = nn.Linear(64, n_cls)

        self.classify = None

    def set_classifier(self, classify):
        self.classify = classify

    def _assign_n_layers(self, rho_list):
        """Map recoverability factor rho -> stopping layer lambda in {2,...,L}.

        Faithful to the official ASDN formula. The result is clamped to
        [2, L] so that out-of-range classifier outputs (e.g. an untrained
        classifier early in training) still land in one of the handled groups
        instead of dropping/corrupting those samples.
        """
        L = self.L
        gamma = self.gamma
        result = []
        for rho in rho_list:
            rho_val = rho.item() if hasattr(rho, 'item') else float(rho)
            n = math.ceil(L - (L - 1) * math.log(gamma * rho_val + 1))
            n = max(2, min(L, n))
            result.append(n)
        return result

    def execute(self, p, x, o, classify_net=None):
        # Faithful port of the official ASDN FeatureExtraction.forward.
        # `batch_size` stays the *original* batch size B throughout; sample
        # subsets are sliced out via `p.reshape(B, o0, -1)[idx]` (NOT by
        # indexing an already-flattened 2D tensor) and merged back afterwards.
        flag = False
        lambda_layer = 4
        L = self.L

        batch_size = p.shape[0]
        num_points = p.shape[1]

        p_from_encoder = []
        x_from_encoder = []
        o_from_encoder = []

        clf = classify_net if classify_net is not None else self.classify
        if clf is not None:
            with jt.no_grad():
                rho_list, _ = clf(p.reshape(batch_size, num_points, -1), None)
            n_layers = self._assign_n_layers(rho_list)
        else:
            n_layers = [4] * batch_size

        layer_2 = [i for i, n in enumerate(n_layers) if n == 2]
        layer_3 = [i for i, n in enumerate(n_layers) if n == 3]
        layer_4 = [i for i, n in enumerate(n_layers) if n == 4]

        # ---- Encoder ----
        for block_i, block in enumerate(self.encoder_blocks):
            o0 = int(o[0].item())
            if block_i == 3 and layer_2:
                if layer_3 or layer_4:
                    keep_idx = jt.array(layer_3 + layer_4).int32()
                    p = p.reshape(batch_size, o0, -1)[keep_idx, :, :].reshape(-1, p.shape[-1])
                    x = x.reshape(batch_size, o0, -1)[keep_idx, :, :].reshape(-1, x.shape[-1])
                    o = o[:len(layer_3 + layer_4)]
                else:
                    flag = True

            if block_i == 4 and layer_3:
                if layer_4:
                    if layer_2:
                        p = p[len(layer_3) * o0:, :]
                        x = x[len(layer_3) * o0:, :]
                        o = o[:len(layer_4)]
                    else:
                        keep_idx = jt.array(layer_4).int32()
                        p = p.reshape(batch_size, o0, -1)[keep_idx, :, :].reshape(-1, p.shape[-1])
                        x = x.reshape(batch_size, o0, -1)[keep_idx, :, :].reshape(-1, x.shape[-1])
                        o = o[:len(layer_4)]
                else:
                    flag = True

            if not flag:
                p, x, o = block(p.reshape(-1, 3), x, o)
                p_from_encoder.append(p.reshape(-1, 3))
                x_from_encoder.append(x)
                o_from_encoder.append(o)
            else:
                lambda_layer = block_i - 1
                break

        # ---- Prepare for decoder ----
        # Initial pop: deepest encoder output (p, x, o already hold this).
        x_dense = x_from_encoder.pop()
        p_dense = p_from_encoder.pop()
        o_dense = o_from_encoder.pop()

        # ---- Decoder ----
        for block_i, block in enumerate(self.decoder_blocks[L - lambda_layer:]):
            # Skip connection (dense, high-res) at this decoder level.
            x_dense = x_from_encoder.pop()
            p_dense = p_from_encoder.pop()
            o_dense = o_from_encoder.pop()
            od0 = int(o_dense[0].item())

            p_dense_corres = p_dense
            x_dense_corres = x_dense
            o_dense_corres = o_dense

            offset = 0 - (L - lambda_layer)
            if block_i == offset and layer_3:
                if layer_2:
                    p_dense_corres = p_dense[len(layer_3) * od0:, :]
                    x_dense_corres = x_dense[len(layer_3) * od0:, :]
                    o_dense_corres = o_dense[:len(layer_4)]
                else:
                    keep_idx = jt.array(layer_4).int32()
                    p_dense_corres = p_dense.reshape(batch_size, od0, -1)[keep_idx, :, :].reshape(-1, p_dense.shape[-1])
                    x_dense_corres = x_dense.reshape(batch_size, od0, -1)[keep_idx, :, :].reshape(-1, x_dense.shape[-1])
                    o_dense_corres = o_dense[:len(layer_4)]

            if block_i == 1 + offset and layer_2:
                keep_idx = jt.array(layer_3 + layer_4).int32()
                p_dense_corres = p_dense.reshape(batch_size, od0, -1)[keep_idx, :, :].reshape(-1, p_dense.shape[-1])
                x_dense_corres = x_dense.reshape(batch_size, od0, -1)[keep_idx, :, :].reshape(-1, x_dense.shape[-1])
                o_dense_corres = o_dense[:len(layer_3 + layer_4)]

            bsize = o_dense_corres.shape[0]
            # p_dense_corres = skip (dense/high-res) -> x1; p = sparse/low-res -> x2.
            p, x, o = block(
                p_dense_corres, x_dense_corres, o_dense_corres,
                p.reshape(-1, 3), x, o,
                batch_size=bsize,
            )

            # Scatter the (partial) deep results back into the full dense batch,
            # keeping early-stopped samples' skip features for the rest.
            if p.shape[0] != p_dense.shape[0]:
                if block_i == offset:
                    if layer_2:
                        n_skip = len(layer_3) * od0
                        p = jt.concat([p_dense[:n_skip], p], dim=0)
                        x = jt.concat([x_dense[:n_skip], x], dim=0)
                        o = o_dense
                    else:
                        pd = p_dense.reshape(batch_size, od0, -1).clone()
                        xd = x_dense.reshape(batch_size, od0, -1).clone()
                        keep_idx = jt.array(layer_4).int32()
                        pd[keep_idx, :, :] = p.reshape(len(layer_4), -1, p.shape[-1])
                        xd[keep_idx, :, :] = x.reshape(len(layer_4), -1, x.shape[-1])
                        o = o_dense
                        p = pd.reshape(-1, p.shape[-1])
                        x = xd.reshape(-1, x.shape[-1])

                if block_i == 1 + offset:
                    pd = p_dense.reshape(batch_size, od0, -1).clone()
                    xd = x_dense.reshape(batch_size, od0, -1).clone()
                    keep_idx = jt.array(layer_3 + layer_4).int32()
                    pd[keep_idx, :, :] = p.reshape(len(layer_3 + layer_4), -1, p.shape[-1])
                    xd[keep_idx, :, :] = x.reshape(len(layer_3 + layer_4), -1, x.shape[-1])
                    o = o_dense
                    p = pd.reshape(-1, p.shape[-1])
                    x = xd.reshape(-1, x.shape[-1])

        # ---- Final MLP ----
        x = nn.relu(self.linear0_1(x))
        x = nn.relu(self.linear0_2(x))
        x_out = jt.tanh(self.linear0_3(x))
        x_out = x_out.reshape(batch_size, num_points, -1)

        return x_out


_knn_proj_cache = {}

def _get_knn_proj(in_dim):
    if in_dim not in _knn_proj_cache:
        proj = jt.randn(in_dim, 3)
        proj = proj / jt.norm(proj, dim=0, keepdims=True)
        _knn_proj_cache[in_dim] = proj
    return _knn_proj_cache[in_dim]

def get_knn_idx(x, y, k, offset=0):
    """
    x: (B, N, d)
    y: (B, M, d)
    return: (B, N, k)
    """
    K = k + offset
    if x.shape[-1] == 3:
        _, idx = jt.misc.knn(x, y, K)
    else:
        proj = _get_knn_proj(x.shape[-1])
        x_proj = x @ proj
        y_proj = y @ proj
        _, idx = jt.misc.knn(x_proj, y_proj, K)
    return idx[:, :, offset:]
