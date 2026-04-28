import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def save_primal_dual_visualizations(primal_feat, dual_feat, sample_idx=0, num_channels=3, save_dir="vis/primal_dual", prefix="sample"):
    os.makedirs(save_dir, exist_ok=True)
    
    primal = primal_feat[sample_idx][:num_channels]  # shape: [C, H, W]
    dual = dual_feat[sample_idx][:num_channels]      # shape: [C, H-1, W-1]

    # Upsample dual to match primal resolution
    dual_upsampled = F.interpolate(dual.unsqueeze(0), size=primal.shape[1:], mode='bilinear', align_corners=False).squeeze(0)

    for i in range(num_channels):
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        
        axs[0].imshow(TF.to_pil_image(primal[i].detach().cpu()))
        axs[0].set_title(f'Primal Channel {i}')
        axs[0].axis('off')
        
        axs[1].imshow(TF.to_pil_image(dual_upsampled[i].detach().cpu()))
        axs[1].set_title(f'Dual Channel {i} (upsampled)')
        axs[1].axis('off')
        
        filename = os.path.join(save_dir, f"{prefix}_ch{i}.png")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)
        # print(f"Saved: {filename}")

def save_line_field_visualizations(h_fw, h_bw, v_fw, v_bw, sample_idx=0, save_dir="vis/line_fields", prefix="sample"):
    os.makedirs(save_dir, exist_ok=True)

    def to_numpy(t):
        return t[sample_idx].squeeze(0).detach().cpu().numpy()

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].imshow(to_numpy(h_fw), cmap='gray')
    axs[0, 0].set_title('Horizontal Forward (v)')
    axs[0, 1].imshow(to_numpy(h_bw), cmap='gray')
    axs[0, 1].set_title('Horizontal Backward (v)')

    axs[1, 0].imshow(to_numpy(v_fw), cmap='gray')
    axs[1, 0].set_title('Vertical Forward (ℓ)')
    axs[1, 1].imshow(to_numpy(v_bw), cmap='gray')
    axs[1, 1].set_title('Vertical Backward (ℓ)')

    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{prefix}_linefields.png")
    plt.savefig(save_path)
    plt.close(fig)
    # print(f"Saved: {save_path}")


# def create_grid_edges(H, W, device):
#     idx = torch.arange(H * W, device=device).reshape(H, W)
#     edges = []
#     for i in range(H):
#         for j in range(W):
#             center = idx[i, j].item()
#             for ni, nj in [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]:
#                 if 0 <= ni < H and 0 <= nj < W:
#                     neighbor = idx[ni, nj].item()
#                     edges.append([center, neighbor])
#     edge_index = torch.tensor(edges, device=device).t().contiguous()
#     return edge_index

def build_grid_graph(feat_map):
    """
    Convert BxCxHxW tensor into list of PyG graph Data objects.
    """
    B, C, H, W = feat_map.shape
    device = feat_map.device  # get current device
    graphs = []

    for b in range(B):
        x = feat_map[b].permute(1, 2, 0).reshape(-1, C)  # [H*W, C]

        idx = torch.arange(H * W, device=device).reshape(H, W)
        edges = []

        for i in range(H):
            for j in range(W):
                center = idx[i, j].item()
                for ni, nj in [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]:
                    if 0 <= ni < H and 0 <= nj < W:
                        neighbor = idx[ni, nj].item()
                        edges.append([center, neighbor])

        edge_index = torch.tensor(edges, device=device).t().contiguous()
        data = Data(x=x, edge_index=edge_index)
        graphs.append(data)

    return Batch.from_data_list(graphs).to(device)

# def build_grid_graph(feat_map, edge_index_cache=None):
#     B, C, H, W = feat_map.shape
#     device = feat_map.device

#     if edge_index_cache is None:
#         edge_index_cache = create_grid_edges(H, W, device)

#     graphs = []
#     for b in range(B):
#         x = feat_map[b].permute(1, 2, 0).reshape(-1, C)  # [H*W, C]
#         data = Data(x=x, edge_index=edge_index_cache)
#         graphs.append(data)

#     return Batch.from_data_list(graphs).to(device)


class GraphConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gconv = GCNConv(in_channels, out_channels)
    
    def forward(self, feat_map):
        B, C, H, W = feat_map.shape
        batch_graph = build_grid_graph(feat_map)
        out = self.gconv(batch_graph.x, batch_graph.edge_index)
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        return out

class ConvBlock(nn.Module):
    """Basic ConvBlock used as backbone."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DualLatticeUpdate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fuse_primal_to_dual = nn.Conv2d(channels, channels, kernel_size=1)
        self.fuse_dual_to_primal = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, F_primal, F_dual):
        F_dual_up = F.interpolate(F_dual, size=F_primal.shape[-2:], mode='bilinear', align_corners=False)
        F_primal_up = F.interpolate(F_primal, size=F_dual.shape[-2:], mode='bilinear', align_corners=False)

        updated_primal = F_primal + self.fuse_dual_to_primal(F_dual_up)
        updated_dual = F_dual + self.fuse_primal_to_dual(F_primal_up)

        return updated_primal, updated_dual

# class DualLatticeUpdate(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.dual_to_primal = GraphConvBlock(channels, channels)
#         self.primal_to_dual = GraphConvBlock(channels, channels)

#     def forward(self, F_primal, F_dual, t=None):
#         F_dual_up = F.interpolate(F_dual, size=F_primal.shape[-2:], mode='bilinear', align_corners=False)
#         F_primal_up = F.interpolate(F_primal, size=F_dual.shape[-2:], mode='bilinear', align_corners=False)

#         # Apply GraphConv-based message passing
#         delta_primal = self.dual_to_primal(F_dual_up)
#         delta_dual = self.primal_to_dual(F_primal_up)

#         if t is not None:
#             cond = t.view(-1, 1, 1, 1)  # broadcast for diffusion timestep
#             delta_primal += cond
#             delta_dual += cond

#         F_primal = F_primal + delta_primal
#         F_dual = F_dual + delta_dual
#         return F_primal, F_dual


def extract_dual_features(F):
    B, C, H, W = F.shape
    dual_feats = F.unfold(2, 2, 1).unfold(3, 2, 1)  # [B, C, H-1, W-1, 2, 2]
    dual_feats = dual_feats.contiguous().view(B, C, 4, (H - 1) * (W - 1))
    dual_feats = dual_feats.mean(dim=2)  # average 2x2 clique
    return dual_feats.view(B, C, H - 1, W - 1)

class DualAwareFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, feat_channels=64, num_blocks=3):
        super().__init__()
        self.init = ConvBlock(in_channels, feat_channels)
        self.blocks = nn.Sequential(*[ConvBlock(feat_channels, feat_channels) for _ in range(num_blocks)])
        self.dual_update = DualLatticeUpdate(channels=feat_channels)

    def forward(self, x):
        # Primal feature map
        F_primal = self.init(x)
        F_primal = self.blocks(F_primal)
        
        # Extract dual features
        F_dual = extract_dual_features(F_primal)

        # Perform dual-primal interaction
        F_primal, F_dual = self.dual_update(F_primal, F_dual)
        # print(np.shape(F_primal), np.shape(F_dual))
        return {
            'primal': F_primal,  # [B, C, H, W]
            'dual': F_dual,      # [B, C, H-1, W-1]
        }

# class DualLatticeMRF(nn.Module):
#     def __init__(self, in_channels, primal_out, dual_out):
#         super().__init__()
#         self.extractor = DualAwareFeatureExtractor(in_channels, primal_out, dual_out)

#     def forward(self, x):
#         #primal, dual = self.extractor(x)
#         output = self.extractor(x)
#         primal = output["primal"]
#         dual = output["dual"]
#         return self.compute_energy(primal, dual)

#     def compute_energy(self, primal_feat, dual_feat):
#         if primal_feat.shape[-2:] != dual_feat.shape[-2:]:
#             dual_feat = F.interpolate(dual_feat, size=primal_feat.shape[-2:], mode='bilinear', align_corners=False)
        
#         fused = torch.cat([primal_feat, dual_feat], dim=1)
#         patch_energy = F.unfold(fused, kernel_size=3, padding=1).pow(2).mean(dim=1)
#         return patch_energy.mean(dim=1)


class DualLatticeMRF(nn.Module):
    def __init__(self, in_channels, primal_out, dual_out, threshold=0.05, lambda_reg=0.1, use_mrf_energy=True):
        super().__init__()
        self.extractor = DualAwareFeatureExtractor(in_channels, primal_out, dual_out)
        #self.threshold = threshold
        self.lambda_reg = lambda_reg
        self.use_mrf_energy = use_mrf_energy  # toggle MRF component
        
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.sharpness = nn.Parameter(torch.tensor(100.0))  

    def forward(self, x):
        output = self.extractor(x)
        primal = output["primal"]
        dual = output["dual"]
        save_primal_dual_visualizations(
            primal_feat=output['primal'],
            dual_feat=output['dual'],
            sample_idx=0,
            num_channels=3,
            save_dir="outputs_1/visualizations/primal_dual",
            prefix="epoch_10"
        )
        # return self.compute_energy(primal, dual)
        return self.compute_energy(primal, dual, x)

    # def compute_energy(self, primal_feat, dual_feat):
    #     if primal_feat.shape[-2:] != dual_feat.shape[-2:]:
    #         dual_feat = F.interpolate(dual_feat, size=primal_feat.shape[-2:], mode='bilinear', align_corners=False)
        
    #     # === Learned patch-based energy ===
    #     fused = torch.cat([primal_feat, dual_feat], dim=1)
    #     patch_energy = F.unfold(fused, kernel_size=3, padding=1).pow(2).mean(dim=1)  # [B, H*W]
    #     patch_energy = patch_energy.mean(dim=1)  # [B]

    #     if self.use_mrf_energy:
    #         mrf_energy, h_edge_fw, h_edge_bw, v_edge_fw, v_edge_bw = self._compute_mrf_energy(primal_feat, return_edges=True)

    #         # Visualize line fields
    #         save_line_field_visualizations(
    #             h_edge_fw, h_edge_bw, v_edge_fw, v_edge_bw,
    #             sample_idx=0,
    #             save_dir="outputs/visualizations/line_fields",
    #             prefix="epoch_10"
    #         )

    #     # if self.use_mrf_energy:
    #     #     # === Explicit MRF energy over primal ===
    #     #     mrf_energy = self._compute_mrf_energy(primal_feat)
    #         total_energy = patch_energy + mrf_energy
    #         return total_energy
    #     else:
    #         return patch_energy

    def compute_energy(self, primal_feat, dual_feat, rgb_input):
        if primal_feat.shape[-2:] != dual_feat.shape[-2:]:
            dual_feat = F.interpolate(dual_feat, size=primal_feat.shape[-2:], mode='bilinear', align_corners=False)
        
        # === Learned patch-based energy ===
        fused = torch.cat([primal_feat, dual_feat], dim=1)
        patch_energy = F.unfold(fused, kernel_size=3, padding=1).pow(2).mean(dim=1)  # [B, H*W]
        patch_energy = patch_energy.mean(dim=1)  # [B]

        if self.use_mrf_energy:
            mrf_energy, edge_maps = self._compute_mrf_energy(rgb_input, return_edges=True)
            h_edge_fw, h_edge_bw, v_edge_fw, v_edge_bw = edge_maps[0]  # Visualize only channel 0 (R)

            save_line_field_visualizations(
                h_edge_fw, h_edge_bw, v_edge_fw, v_edge_bw,
                sample_idx=0,
                save_dir="outputs_1/visualizations/line_fields",
                prefix="epoch_10"
            )

            total_energy = patch_energy + mrf_energy
            return total_energy
        else:
            return patch_energy

    # def _compute_mrf_energy(self, x):
    #     # x: [B, C, H, W] → convert to grayscale-like field if needed
    #     if x.shape[1] > 1:
    #         x = x.mean(dim=1, keepdim=True)

    #     h_diff = x[:, :, :, :-1] - x[:, :, :, 1:]  # Horizontal
    #     v_diff = x[:, :, :-1, :] - x[:, :, 1:, :]  # Vertical

    #     h_edge = (h_diff.abs() > self.threshold).float()
    #     v_edge = (v_diff.abs() > self.threshold).float()
    #     # print('h edge, v edge')
    #     # print(h_edge, v_edge)
    #     h_term = ((h_diff ** 2) * (1 - h_edge)).mean()
    #     v_term = ((v_diff ** 2) * (1 - v_edge)).mean()
    #     reg_term = self.lambda_reg * (h_edge.mean() + v_edge.mean())

    #     return h_term + v_term + reg_term

    # def _compute_mrf_energy(self, x, return_edges =False): # 1 channel
    #     # Convert to grayscale if needed
    #     if x.shape[1] > 1:
    #         x = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]

    #     # Forward horizontal and vertical diffs
    #     h_diff_fw = x[:, :, :, :-1] - x[:, :, :, 1:]    # x_{i,j} - x_{i,j+1}
    #     v_diff_fw = x[:, :, :-1, :] - x[:, :, 1:, :]    # x_{i,j} - x_{i+1,j}

    #     # Backward horizontal and vertical diffs
    #     h_diff_bw = x[:, :, :, 1:] - x[:, :, :, :-1]    # x_{i,j+1} - x_{i,j}
    #     v_diff_bw = x[:, :, 1:, :] - x[:, :, :-1, :]    # x_{i+1,j} - x_{i,j}

    #     # Edge masks (binary line fields) using threshold
    #     h_edge_fw = (h_diff_fw.abs() > self.threshold).float()
    #     v_edge_fw = (v_diff_fw.abs() > self.threshold).float()
    #     h_edge_bw = (h_diff_bw.abs() > self.threshold).float()
    #     v_edge_bw = (v_diff_bw.abs() > self.threshold).float()

    #     h_edge_fw = torch.sigmoid((h_diff_fw.abs() - self.threshold) * self.sharpness)
    #     v_edge_fw = torch.sigmoid((v_diff_fw.abs() - self.threshold) * self.sharpness)
    #     h_edge_bw = torch.sigmoid((h_diff_bw.abs() - self.threshold) * self.sharpness)
    #     v_edge_bw = torch.sigmoid((v_diff_bw.abs() - self.threshold) * self.sharpness)


    #     # Forward and backward energy terms
    #     h_term_fw = ((h_diff_fw ** 2) * (1 - h_edge_fw)).mean()
    #     v_term_fw = ((v_diff_fw ** 2) * (1 - v_edge_fw)).mean()
    #     h_term_bw = ((h_diff_bw ** 2) * (1 - h_edge_bw)).mean()
    #     v_term_bw = ((v_diff_bw ** 2) * (1 - v_edge_bw)).mean()

    #     # Regularization encourages edge sparsity
    #     reg_term = self.lambda_reg * (
    #         h_edge_fw.mean() + h_edge_bw.mean() +
    #         v_edge_fw.mean() + v_edge_bw.mean()
    #     )

    #     if return_edges:
    #         return (
    #             h_term_fw + h_term_bw + v_term_fw + v_term_bw + reg_term,
    #             h_edge_fw, h_edge_bw,
    #             v_edge_fw, v_edge_bw
    #         )
    #     else:
    #         return h_term_fw + h_term_bw + v_term_fw + v_term_bw + reg_term

    #     # return h_term_fw + h_term_bw + v_term_fw + v_term_bw + reg_term

    # def _compute_mrf_energy(self, x, return_edges=False): ## 3 channels 
    #     # x: [B, C, H, W], where C=3 for RGB
    #     energies = []
    #     edge_maps = []  # only used if return_edges=True

    #     for c in range(x.shape[1]):  # Loop over each channel
    #         x_c = x[:, c:c+1, :, :]  # Isolate channel c

    #         # Forward horizontal and vertical diffs
    #         h_diff_fw = x_c[:, :, :, :-1] - x_c[:, :, :, 1:]
    #         v_diff_fw = x_c[:, :, :-1, :] - x_c[:, :, 1:, :]

    #         # Backward horizontal and vertical diffs
    #         h_diff_bw = x_c[:, :, :, 1:] - x_c[:, :, :, :-1]
    #         v_diff_bw = x_c[:, :, 1:, :] - x_c[:, :, :-1, :]

    #         # Edge masks using soft thresholds (sigmoid smoothing)
    #         h_edge_fw = torch.sigmoid((h_diff_fw.abs() - self.threshold) * self.sharpness)
    #         v_edge_fw = torch.sigmoid((v_diff_fw.abs() - self.threshold) * self.sharpness)
    #         h_edge_bw = torch.sigmoid((h_diff_bw.abs() - self.threshold) * self.sharpness)
    #         v_edge_bw = torch.sigmoid((v_diff_bw.abs() - self.threshold) * self.sharpness)

    #         # Forward and backward energy terms
    #         h_term_fw = ((h_diff_fw ** 2) * (1 - h_edge_fw)).mean()
    #         v_term_fw = ((v_diff_fw ** 2) * (1 - v_edge_fw)).mean()
    #         h_term_bw = ((h_diff_bw ** 2) * (1 - h_edge_bw)).mean()
    #         v_term_bw = ((v_diff_bw ** 2) * (1 - v_edge_bw)).mean()

    #         # Regularization for edge sparsity
    #         reg_term = self.lambda_reg * (
    #             h_edge_fw.mean() + h_edge_bw.mean() +
    #             v_edge_fw.mean() + v_edge_bw.mean()
    #         )

    #         total_energy = h_term_fw + v_term_fw + h_term_bw + v_term_bw + reg_term
    #         energies.append(total_energy)

    #         if return_edges:
    #             edge_maps.append((h_edge_fw, h_edge_bw, v_edge_fw, v_edge_bw))

    #     total_energy = torch.stack(energies).mean()  # or .sum() if you prefer stronger penalty

    #     if return_edges:
    #         # edge_maps is a list of tuples per channel; you can return the full list or merge as needed
    #         return total_energy, edge_maps
    #     else:
    #         return total_energy

    def _compute_mrf_energy(self, x, return_edges=False):
        # x: [B, 3, H, W] — RGB image
        B, C, H, W = x.shape
        assert C == 3, "Input must have 3 channels (RGB)"

        intra_energies = []
        inter_energies = []
        edge_maps = []  # Only used if return_edges=True

        # ---------- Intra-Channel Energy (within each color plane) ----------
        for c in range(C):
            x_c = x[:, c:c+1, :, :]

            # Forward differences
            h_diff_fw = x_c[:, :, :, :-1] - x_c[:, :, :, 1:]
            v_diff_fw = x_c[:, :, :-1, :] - x_c[:, :, 1:, :]

            # Backward differences
            h_diff_bw = x_c[:, :, :, 1:] - x_c[:, :, :, :-1]
            v_diff_bw = x_c[:, :, 1:, :] - x_c[:, :, :-1, :]

            # Soft edge masks
            h_edge_fw = torch.sigmoid((h_diff_fw.abs() - self.threshold) * self.sharpness)
            v_edge_fw = torch.sigmoid((v_diff_fw.abs() - self.threshold) * self.sharpness)
            h_edge_bw = torch.sigmoid((h_diff_bw.abs() - self.threshold) * self.sharpness)
            v_edge_bw = torch.sigmoid((v_diff_bw.abs() - self.threshold) * self.sharpness)

            # Energy terms
            h_term_fw = ((h_diff_fw ** 2) * (1 - h_edge_fw)).mean()
            v_term_fw = ((v_diff_fw ** 2) * (1 - v_edge_fw)).mean()
            h_term_bw = ((h_diff_bw ** 2) * (1 - h_edge_bw)).mean()
            v_term_bw = ((v_diff_bw ** 2) * (1 - v_edge_bw)).mean()

            # Edge sparsity regularization
            reg_term = self.lambda_reg * (
                h_edge_fw.mean() + h_edge_bw.mean() +
                v_edge_fw.mean() + v_edge_bw.mean()
            )

            intra_energy = h_term_fw + v_term_fw + h_term_bw + v_term_bw + reg_term
            intra_energies.append(intra_energy)

            if return_edges:
                edge_maps.append((h_edge_fw, h_edge_bw, v_edge_fw, v_edge_bw))

        # ---------- Inter-Channel Energy (across color planes) ----------
        # Define all pairs (R,G), (G,B), (B,R)
        pairs = [(0, 1), (1, 2), (2, 0)]
        for c1, c2 in pairs:
            x1 = x[:, c1:c1+1, :, :]
            x2 = x[:, c2:c2+1, :, :]

            # Forward differences between channels (same pixel location)
            h_diff_fw = x1[:, :, :, :-1] - x2[:, :, :, 1:]
            v_diff_fw = x1[:, :, :-1, :] - x2[:, :, 1:, :]

            # Backward differences
            h_diff_bw = x2[:, :, :, 1:] - x1[:, :, :, :-1]
            v_diff_bw = x2[:, :, 1:, :] - x1[:, :, :-1, :]

            # Soft edge masks
            h_edge_fw = torch.sigmoid((h_diff_fw.abs() - self.threshold) * self.sharpness)
            v_edge_fw = torch.sigmoid((v_diff_fw.abs() - self.threshold) * self.sharpness)
            h_edge_bw = torch.sigmoid((h_diff_bw.abs() - self.threshold) * self.sharpness)
            v_edge_bw = torch.sigmoid((v_diff_bw.abs() - self.threshold) * self.sharpness)

            # Energy terms for inter-plane
            h_term_fw = ((h_diff_fw ** 2) * (1 - h_edge_fw)).mean()
            v_term_fw = ((v_diff_fw ** 2) * (1 - v_edge_fw)).mean()
            h_term_bw = ((h_diff_bw ** 2) * (1 - h_edge_bw)).mean()
            v_term_bw = ((v_diff_bw ** 2) * (1 - v_edge_bw)).mean()

            # Regularization
            reg_term = self.lambda_reg * (
                h_edge_fw.mean() + h_edge_bw.mean() +
                v_edge_fw.mean() + v_edge_bw.mean()
            )

            inter_energy = h_term_fw + v_term_fw + h_term_bw + v_term_bw + reg_term
            inter_energies.append(inter_energy)

        # Combine total energy
        total_energy = (torch.stack(intra_energies).mean() +
                        torch.stack(inter_energies).mean())

        if return_edges:
            return total_energy, edge_maps
        else:
            return total_energy
