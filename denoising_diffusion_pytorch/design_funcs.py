import torch as tr
from torch_geometric.data import Data, Batch
from torch.fft import fftshift, fft, ifft, fftfreq
from scipy.constants import speed_of_light
from scipy.signal.windows import taylor
import numpy as np
from rfsim.coordinate_conversions import ar2los
import matplotlib.pyplot as plt


def parametric_cylinder(r = tr.tensor(1.0), n = tr.tensor(20)):
    r = tr.atleast_1d(r)
    m = len(r)
    if m == 1:
        r = tr.asarray([r, r])[:, None]
        m = 2
    else:
        r = r[:, None]

    θ = tr.arange(n + 1) / n * 2 * tr.pi
    st = tr.sin(θ)
    st[-1] = 0

    x = r * tr.cos(θ)[None]
    y = r * st[None]
    z = (tr.arange(m) / (m - 1))[:, None] * tr.ones((1, n + 1))
    
    return x, y, z

def frusta_collection(
    radius,
    axial,
    top_sphere_length=None,
    bot_sphere_length=None,
    N: int = 20,
    M: int = 6,
):
    # z axis for coordinates is axis of symmetry
    assert len(radius) == len(axial)
    bot_axial = axial[0]
    top_axial = axial[-1]
    coords = []
    faces = []
    for k in range(len(axial) - 1):
        length = axial[k + 1] - axial[k]
        bottom_radius = radius[k]
        top_radius = radius[k + 1]
        xi, yi, zi = parametric_cylinder(radius, N)
        xi = xi[:, :N]
        yi = yi[:, :N]
        zi = zi[:, :N]

        zi = zi * length
        stacked = tr.stack((xi, yi, zi), dim=2).view(-1, 3)
        # add ending 2 points
        coord = tr.cat((tr.tensor([[0, 0, 0]]), stacked, tr.tensor([[0, 0, top_axial]])))
        coords.extend(coord)

        # faces don't need gradient as of now
        fac = []
        if k == 0:
            # Create back cylinder base
            for i in range(1, N):
                fac.append([i, 0, i + 1])
            fac.append([N, 0, 1])

        # Create frustum
        for i in range(1, N):
            fac.append([N + i, i, i + 1])
            fac.append([N + i, i + 1, N + i + 1])
        fac.append([2 * N, N, 1])
        fac.append([2 * N, 1, N + 1])

        if k == len(radius) - 2:
            # Create top
            for i in range(1, N):
                fac.append([2 * N + 1, N + i, N + i + 1])
            fac.append([2 * N + 1, N + N, N + 1])

        num = 2 * N + 2
        fac = tr.tensor(fac) + k * num

        faces.extend(fac)
    
    return tr.stack(faces), tr.stack(coords)


def create(length, nose_radius, base_radius):
    # TODO: make sure gradients are correctly applied when we decide to change other params
    # return frusta_collection(tr.cat([tr.tensor([nose_radius]), tr.tensor([base_radius])]), tr.cat([tr.tensor([0]), length]))
    
    # length = (length - 0.1) / 4.9
    # bound1, bound2 = (length / 2).item(), (2 * length / 3).item()
    # center = (bound1 - bound2) * tr.rand(1) + bound2
    
    zs = tr.cat([tr.tensor([0]), length])
    rs = tr.cat([nose_radius, base_radius])
    faces, vertices = frusta_collection(rs, zs, N=100, M=100)
    center = length / 2
    # shift on Z axis to center
    vertices[:, 2] = vertices[:, 2] - center
    return faces, vertices, zs, rs

# length = nn.Parameter(tr.tensor([2.0], dtype=tr.float32), requires_grad=True)
# create(length, 0.1, 0.1)[0]

def design(inp_len, nose_radius, base_radius, return_edge=False):
    faces, vertices, _, _ = create(inp_len, nose_radius, base_radius)
    # taken from trimesh function
    face_test = faces
    edge1 = faces.reshape(1, -1)
    edge2 = faces[:, [1, 2, 0]].reshape(1, -1)
    edges = tr.cat((edge1, edge2))
    edge_vec = vertices[edges[0]] - vertices[edges[1]]
    
    sample_d = Data(
      pos=vertices,
      x=tr.ones(len(vertices), 1),
      edge_index=edges,
      edge_vec=edge_vec
    )
    batched = Batch.from_data_list([sample_d])
    if return_edge:
        return batched, faces, vertices, edge_vec
    
    return batched, faces, vertices

# design(nn.Parameter(tr.tensor([2.0], dtype=tr.float32, requires_grad=True)))

def rew(rem_mod, pred_latent, target, objective):
    # get RTI
    aspects = tr.linspace(0, np.pi, 360)
    num_samples = 1
    range_profiles = []
    # repeat each aspect for each element in batch
    for i, aspect in enumerate(aspects):
        aspect_batch_repeated = aspect.repeat(num_samples)
        rolls = tr.zeros_like(aspect_batch_repeated)
        ar = tr.stack([aspect_batch_repeated, rolls], -1)
        los = ar2los(ar)
        response_out = rem_mod.getResponse(pred_latent, los) # (num_samples, R)
        range_profiles.append(response_out.unsqueeze(1))

    rti_predicted = tr.cat(range_profiles, dim=1).T
    
    if objective == "mse":
        return mse_loss(rti_predicted, target)
    elif objective == "peak":
        val_diff, _ = peak_match(target, rti_predicted) 
        return val_diff
    elif objective == "peak_bin":
        _, bin_diff = peak_match(target, rti_predicted) 
        return bin_diff
    elif objective == "weighted_comb":
        val_diff, bin_diff = peak_match(target, rti_predicted)
        mse = (1/tr.numel(rti_predicted)) * sum(sum((rti_predicted - target)**2 * target))
        return 0.5*mse + 0.5*val_diff
    elif objective == "freq":
        # assume we apply window on each one
        # do inverse fft and then taylor window, then compute MSE
        rr = tr.linspace(-20, 20, 258)
        freqs_pred = fftshift(fftfreq(rti_predicted.shape[-1], d=rr[1] - rr[0]), -1) * speed_of_light / 2
        fai_pred = ifft(fftshift(rti_predicted, -1), axis=-1)
        weights_pred = tr.tensor(taylor(fai_pred.shape[-1], nbar=7, sll=40), dtype=tr.float32)
        fai_pred = fai_pred / weights_pred[None]
        
        freqs_targ = fftshift(fftfreq(target.shape[-1], d=rr[1] - rr[0]), -1) * speed_of_light / 2
        fai_targ = ifft(fftshift(target, -1), axis=-1)
        weights_targ = tr.tensor(taylor(fai_targ.shape[-1], nbar=7, sll=40), dtype=tr.float32)
        fai_targ = fai_targ / weights_targ[None]
        
        return tr.mean(tr.abs(fai_pred - fai_targ))
    else:
        raise ValueError()
        
def rti_from_latent(rem_mod, pred_latent):
    
    # get RTI
    aspects = tr.linspace(0, np.pi, 360)
    num_samples = 1
    range_profiles = []
    # repeat each aspect for each element in batch
    for i, aspect in enumerate(aspects):
        aspect_batch_repeated = aspect.repeat(num_samples)
        rolls = tr.zeros_like(aspect_batch_repeated)
        ar = tr.stack([aspect_batch_repeated, rolls], -1)
        los = ar2los(ar)
        response_out = rem_mod.getResponse(pred_latent, los.unsqueeze(0)) # (num_samples, R)
        range_profiles.append(response_out.unsqueeze(1))

    rti_predicted = tr.cat(range_profiles, dim=1).T
    return rti_predicted


def viz(rem_mod, pred_latent):
    rti_predicted_viz = rti_from_latent(rem_mod, pred_latent)
    # rti_predicted_scaled = 20 * np.log10(np.abs(rti_predicted_viz) + 1e-10)
    ranges = np.linspace(0, 107*0.3747, 107)
    ranges = ranges - np.median(ranges)
    a = plt.pcolormesh(ranges, aspects, rti_predicted_viz.detach().numpy())
    # plt.xlim((-5, 5))
    plt.xlabel("Relative Range (m)")
    plt.ylabel("Aspect (rad)")
    plt.colorbar()
    
    return rti_predicted_viz.detach()

def viz_rti(pred_rti):
    aspects = tr.linspace(0, np.pi, 360)
    ranges = np.linspace(0, pred_rti.size(1)*0.31, pred_rti.size(1))
    ranges -= np.mean(ranges)
    a = plt.pcolormesh(ranges, aspects, np.log10(pred_rti.detach().cpu().numpy()))
    # plt.xlim((-5, 5))
    plt.xlabel("Relative Range (m)")
    plt.ylabel("Aspect (rad)")
    plt.colorbar()
    