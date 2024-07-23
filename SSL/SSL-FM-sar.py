import torch
import numpy as np
import math
import itertools
import torch.nn.functional as F
from torch import distributed as dist
from torch import nn, einsum
from einops import rearrange
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import optuna
import matplotlib.pyplot as plt
import torch.distributed as dist
import random
import torch
import numpy as np
import math
import itertools
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt



class CROMA(nn.Module):
    def __init__(self,
                 patch_size=8,
                 encoder_dim=768,
                 encoder_layers=12,
                 attention_heads=16,
                 decoder_dim=512,
                 decoder_layers=1,
                 total_channels=3,
                 num_patches=1024,
                 ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.encoder_layers = encoder_layers
        self.decoder_dim = decoder_dim
        self.decoder_layers = decoder_layers
        self.attention_heads = attention_heads
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.total_channels = total_channels
        self.radar_encoder = ViT(num_patches=self.num_patches,
                                          dim=self.encoder_dim,
                                          layers=int(self.encoder_layers/2),
                                          attention_heads=self.attention_heads,
                                          in_channels=3,
                                          patch_size=self.patch_size,
                                          )

        self.cross_encoder = BaseTransformerCrossAttn(dim=self.encoder_dim,
                                                      layers=int(self.encoder_layers/2),
                                                      attention_heads=self.attention_heads,
                                                      )
        self.GAP_FFN_radar = nn.Sequential(
            nn.LayerNorm(self.encoder_dim),
            nn.Linear(self.encoder_dim, int(4*self.encoder_dim)),
            nn.GELU(),
            nn.Linear(int(4*self.encoder_dim), self.encoder_dim)
        )

        self.decoder = DecoderMAE(num_patches=self.num_patches,
                                              encoder_dim=self.encoder_dim,
                                              decoder_dim=self.decoder_dim,
                                              decoder_layers=self.decoder_layers,
                                              attention_heads=8,
                                              total_channels=self.total_channels,
                                              patch_size=self.patch_size,
                                              )
        self.attn_bias = get_alibi(attention_heads=self.attention_heads,
                                               num_patches=self.num_patches)
        self.global_contrast_loss = ContrastLossInput(projection_input=self.encoder_dim,
                                                                  projection_output=self.encoder_dim,
                                                                  )

    def forward(self, imgs, radar_mask_info):
        # split stacked image into optical and radar
        radar_imgs = imgs
        

        # create independent random masks
        radar_masked_attn_bias = apply_mask_to_alibi(alibi=self.attn_bias.to(radar_imgs.device),
                                                              ids_keep_queries=radar_mask_info['ids_keep'],
                                                              ids_keep_keys=radar_mask_info['ids_keep'],
                                                              batch_size=radar_imgs.shape[0],
                                                              orig_seq_len=self.num_patches,
                                                              masked_seq_len=radar_mask_info['len_keep'],
                                                              attention_heads=self.attention_heads)

        # encode each sensor independently
        radar_encodings = self.radar_encoder(imgs=radar_imgs, attn_bias=radar_masked_attn_bias, mask_info=radar_mask_info)
        # create unimodal representations with an FFN
        radar_GAP = self.GAP_FFN_radar(radar_encodings.mean(dim=1))
        # create cross attention bias and create joint multimodal encodings
        cross_attn_bias = apply_mask_to_alibi(alibi=self.attn_bias.to(radar_imgs.device),
                                          ids_keep_queries=radar_mask_info['ids_keep'],
                                          ids_keep_keys=radar_mask_info['ids_keep'],  # Assuming the keys to keep are the same as the queries
                                          batch_size=radar_imgs.shape[0],
                                          orig_seq_len=self.num_patches,
                                          masked_seq_len=radar_mask_info['len_keep'],  # Use the len_keep as masked_seq_len
                                          attention_heads=self.attention_heads)
        # reconstruct both sensors
        patchified_radar_imgs = rearrange(imgs, 'b c (h i) (w j) -> b (h w) (c i j)', i=self.patch_size, j=self.patch_size)
        mae_loss,pred_image = self.decoder(x=radar_encodings, mask_info_radar=radar_mask_info, target=patchified_radar_imgs)
        return mae_loss,pred_image


class FFN(nn.Module):
    def __init__(self,
                 dim,
                 mult=4,
                 ):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim)
        )
        self.input_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.input_norm(x)
        return self.net(x)

class Attention(nn.Module):
    def __init__(self,
                 dim,
                 attention_heads=8,
                 ):
        super().__init__()
        self.attention_heads = attention_heads
        dim_head = int(dim / attention_heads)
        self.scale = dim_head ** -0.5
        self.create_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)
        self.input_norm = nn.LayerNorm(dim)

    def forward(self, x, alibi):
        x = self.input_norm(x)
        q, k, v = self.create_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.attention_heads), (q, k, v))
        attention_scores = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if exists(alibi):
            attention_scores = attention_scores + alibi
        attn = attention_scores.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        return self.out(rearrange(out, 'b h n d -> b n (h d)'))

class CrossAttention(nn.Module):
    def __init__(self,
                 dim,
                 attention_heads=8,
                 ):
        super().__init__()
        self.attention_heads = attention_heads
        dim_head = int(dim / attention_heads)
        self.scale = dim_head ** -0.5
        self.create_q = nn.Linear(dim, dim, bias=False)
        self.create_k = nn.Linear(dim, dim, bias=False)
        self.create_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.input_norm = nn.LayerNorm(dim)

    def forward(self, x, context, alibi):
        x = self.input_norm(x)
        context = self.input_norm(context)
        q = self.create_q(x)
        k = self.create_k(context)
        v = self.create_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.attention_heads), (q, k, v))
        attention_scores = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attention_scores = attention_scores + alibi
        attn = attention_scores.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class BaseTransformer(nn.Module):
    def __init__(self,
                 dim,
                 layers,
                 attention_heads=8,
                 ff_mult=4,
                 final_norm=True,
                 ):
        super().__init__()
        self.final_norm = final_norm
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, attention_heads=attention_heads),
                FFN(dim=dim, mult=ff_mult),
            ]))
        if self.final_norm:
            self.norm_out = nn.LayerNorm(dim)

    def forward(self, x, alibi=None):
        for self_attn, ffn in self.layers:
            x = self_attn(x, alibi) + x
            x = ffn(x) + x
        if self.final_norm:
            return self.norm_out(x)
        else:
            return x

class BaseTransformerCrossAttn(nn.Module):
    def __init__(self,
                 dim,
                 layers,
                 attention_heads=8,
                 ff_mult=4,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, attention_heads=attention_heads),
                CrossAttention(dim=dim, attention_heads=attention_heads),
                FFN(dim=dim, mult=ff_mult),
            ]))
        self.norm_out = nn.LayerNorm(dim)
    def forward(self, x, context, alibi):
        for self_attn, cross_attn, ffn in self.layers:
            x = self_attn(x, alibi) + x
            x = cross_attn(x, context, alibi) + x
            x = ffn(x) + x
        x = self.norm_out(x)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_alibi(attention_heads, num_patches):
    points = list(itertools.product(range(int(math.sqrt(num_patches))), range(int(math.sqrt(num_patches)))))

    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                               :n - closest_power_of_2]
    slopes = torch.Tensor(get_slopes(attention_heads)).unsqueeze(1)
    idxs = []
    for p1 in points:
        for p2 in points:
            dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            idxs.append(dist * slopes * -1)
    all_bias = torch.cat(idxs, dim=1)
    return all_bias.view(1, attention_heads, num_patches, num_patches)


def get_mask(bsz, seq_len, device, mask_ratio):
    len_keep = int(seq_len * (1 - mask_ratio))
    noise = torch.rand(bsz, seq_len, device=device)  # noise in [0, 1]
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    mask = torch.ones([bsz, seq_len], device=device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    mask_info = {
        'ids_restore': ids_restore,
        'ids_keep': ids_keep,
        'len_keep': len_keep,
        'mask_for_mae': mask
    }
    return mask_info


def apply_mask_to_sequence(x, ids_keep):
    return torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))


def apply_mask_to_alibi(alibi, ids_keep_queries, ids_keep_keys, batch_size, orig_seq_len, masked_seq_len,
                        attention_heads):    
    ids_keep_matrix = rearrange(ids_keep_queries, 'b i -> b i 1')\
                      + rearrange(ids_keep_keys, 'b i -> b 1 i') * orig_seq_len
    ids_keep_long_sequence = rearrange(ids_keep_matrix, 'b i j -> b (i j)')
    alibi_long_sequence = rearrange(alibi.repeat(batch_size, 1, 1, 1), 'b n i j -> b (i j) n')
    alibi_masked = torch.gather(alibi_long_sequence, dim=1,
                                index=ids_keep_long_sequence.unsqueeze(-1).repeat(1, 1, attention_heads))    
    return rearrange(alibi_masked, 'b (i j) n -> b n i j', i=masked_seq_len, j=masked_seq_len)


def gather_features(features, world_size):
    gathered_image_features = [torch.zeros_like(features) for _ in range(world_size)]
    dist.all_gather(gathered_image_features, features)
    all_features = torch.cat(gathered_image_features, dim=0)
    return all_features


class ContrastLossInput(nn.Module):
    def __init__(
            self,
            projection_input=768,
            projection_output=768,
    ):
        super().__init__()
        self.radar_proj = nn.Linear(projection_input, projection_output)
        self.optical_proj = nn.Linear(projection_input, projection_output)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, radar_features, optical_features, world_size, rank):
        # linear projection of unimodal representations
        radar_features = self.radar_proj(radar_features)
        optical_features = self.optical_proj(optical_features)

        # L2 normalize
        radar_features = radar_features / radar_features.norm(dim=1, keepdim=True)
        optical_features = optical_features / optical_features.norm(dim=1, keepdim=True)
        # gather features from other GPUs
        all_radar_features = gather_features(features=radar_features, world_size=world_size)
        all_optical_features = gather_features(features=optical_features, world_size=world_size)
        # dot product to get logits
        logit_scale = self.logit_scale.exp()
        logits_per_optical = logit_scale * optical_features @ all_radar_features.t()
        logits_per_radar = logit_scale * radar_features @ all_optical_features.t()
        # organize labels
        num_logits = logits_per_optical.shape[0]
        labels = torch.arange(num_logits, device=radar_features.device, dtype=torch.long)
        labels = labels + num_logits * rank
        # calculate loss
        loss = (F.cross_entropy(logits_per_optical, labels) + F.cross_entropy(logits_per_radar, labels)) / 2
        return loss


class ViT(nn.Module):
    def __init__(self,
                 num_patches,
                 dim=768,
                 layers=12,
                 attention_heads=16,
                 in_channels=3,
                 patch_size=8,
                 ):
        super().__init__()
        self.dim = dim
        self.layers = layers
        self.attention_heads = attention_heads
        self.num_patches = num_patches
        self.patch_size = patch_size
        pixels_per_patch = int(patch_size * patch_size * in_channels)
        self.linear_input = nn.Linear(pixels_per_patch, self.dim)
        self.transformer = BaseTransformer(dim=self.dim,
                                           layers=self.layers,
                                           attention_heads=self.attention_heads,
                                           )

    def forward(self, imgs, attn_bias, mask_info=None):
        x = rearrange(imgs, 'b c (h i) (w j) -> b (h w) (c i j)', i=self.patch_size, j=self.patch_size)
        x = self.linear_input(x)
        if mask_info is None:
            x = self.transformer(x, alibi=attn_bias)
            return x
        else:
            x_masked = apply_mask_to_sequence(x=x, ids_keep=mask_info['ids_keep'])
            x_masked = self.transformer(x_masked, alibi=attn_bias)
            return x_masked
class DecoderMAE(nn.Module):
    def __init__(self,
                 num_patches,
                 encoder_dim=768,
                 decoder_dim=768,
                 decoder_layers=12,
                 attention_heads=16,
                 total_channels=3,
                 patch_size=8,
                 ):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.decoder_layers = decoder_layers
        self.attention_heads = attention_heads
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.encoder_to_decoder = nn.Linear(encoder_dim, self.decoder_dim)
        self.decoder = BaseTransformer(dim=self.decoder_dim,
                                       layers=self.decoder_layers,
                                       attention_heads=self.attention_heads,
                                       )
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.decoder_dim), requires_grad=False)
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(num_patches ** .5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        pixels_per_patch = int(patch_size * patch_size * total_channels)
        self.linear_output = nn.Linear(self.decoder_dim, pixels_per_patch)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)
    def forward(self, x, mask_info_radar, target):
        # Prepare inputs for the decoder
        x = self.encoder_to_decoder(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], mask_info_radar['ids_restore'].shape[1] + 1 - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=mask_info_radar['ids_restore'].unsqueeze(-1).repeat(1, 1, x.shape[2]))

        # Decode embeddings
        x = x + self.decoder_pos_embed
        x = self.linear_output(self.decoder(x))

        # Only reconstruct radar data
        num_patches_per_dim = int((256 / 8))  # This should be 32

        # Now, rearrange the patches back to the original image layout
        pred_radar = rearrange(x, 'b (h w) (c ph pw) -> b c (h ph) (w pw)', 
                                       h=num_patches_per_dim, w=num_patches_per_dim, 
                               ph=8, pw=8, c=3)
        # Apply patch-wise normalization
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6) ** .5
        # Only handle target radar data
        target_radar = rearrange(target, 'b (h w) (c ph pw) -> b c (h ph) (w pw)', 
                                       h=num_patches_per_dim, w=num_patches_per_dim, 
                               ph=8, pw=8, c=3)
        # Calculate radar reconstruction loss
        loss_radar = F.mse_loss(x, target, reduction='none')
        loss_radar = loss_radar.mean(dim=-1)  # [N, L], mean loss per patch
        # Apply mask and compute mean loss on removed patches
        mask = mask_info_radar['mask_for_mae']
        loss_radar = (loss_radar * mask).sum() / (mask.sum() + 1e-8)
        
        

        return loss_radar, pred_radar
    
def exists(val):
    return val is not None


def split_into_patches(data, patch_size=(256, 256)):
    """
    Split image data into 256x256 patches.
    
    Args:
        data (numpy.ndarray): Input image data of shape (channels, height, width).
        patch_size (tuple): The height and width of each patch.
    
    Returns:
        numpy.ndarray: Array of image patches of shape (num_patches, channels, patch_size[0], patch_size[1]).
    """
    channels, height, width = data.shape
    patch_height, patch_width = patch_size
    patches = []
    
    for i in range(0, height, patch_height):
        for j in range(0, width, patch_width):
            if (i + patch_height) <= height and (j + patch_width) <= width:  # Ensure the patch does not exceed image bounds
                patch = data[:, i:i+patch_height, j:j+patch_width]
                patches.append(patch)
    
    return np.array(patches)

def load_and_stack_data(folder_path):
    all_patches = []  # Initialize a list to hold all patches from all images
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".npy"):
            data = np.load(os.path.join(folder_path, file_name))
            data[data < -100] = np.nan
            data[data > 10] = np.nan
            patches = split_into_patches(data[0:3,:,:], patch_size=(256, 256))
            all_patches.extend(patches)
    
    # Stack all patches into a single numpy array
    stacked_patches = np.stack(all_patches, axis=0)
    return stacked_patches

# Usage
folder_path = 'sentinel-1'
stacked_patches = load_and_stack_data(folder_path)
print(f"Shape of stacked patches: {stacked_patches.shape}")


def compute_channel_stats(data):
    """
    Compute mean and standard deviation for each channel in the data.
    data: np.array of shape (samples, channels, height, width)
    """
    means = np.nanmean(data, axis=(0, 2, 3))
    
    mins=np.nanmin(data, axis=(0, 2, 3))
    maxs=np.nanmax(data, axis=(0, 2, 3))
    
    print(mins)
    print(maxs)
    stds = np.nanstd(data, axis=(0, 2, 3))
    return means, stds,mins,maxs

def normalize_data(data, means, stds):
    """
    Normalize data using the provided means and standard deviations for each channel.
    data: np.array of shape (samples, channels, height, width)
    means, stds: np.arrays of shape (channels,)
    """
    # Ensure means and stds are reshaped for broadcasting
    means = means.reshape((1, -1, 1, 1))
    stds = stds.reshape((1, -1, 1, 1))
    normalized_data = (data - means) / stds
    
    return normalized_data

class PatchesDataset(Dataset):
    def __init__(self, patches):
        """
        Args:
            patches (numpy.ndarray): Array of image patches of shape (num_patches, channels, height, width).
        """
        self.patches = patches
        
        # Initialize a list to keep track of valid patch indices
        self.valid_indices = []
        
        # Iterate over all patches and check for NaN values
        for i, patch in enumerate(self.patches):
            if not np.isnan(patch).any():  # Check if the patch contains any NaN values
                self.valid_indices.append(i)  # If not, add the index to valid_indices list

    def __len__(self):
        return len(self.valid_indices)  # Return the number of valid patches

    def __getitem__(self, idx):
        # Use the valid index to get the actual patch index
        valid_idx = self.valid_indices[idx]
        
        # Convert the numpy patch to a torch tensor
        patch = self.patches[valid_idx]
        patch_tensor = torch.tensor(patch, dtype=torch.float)
        return patch_tensor
    
# Example for one year
means, stds,mins,maxs = compute_channel_stats(stacked_patches)
normalized_data = normalize_data(stacked_patches, means, stds)  # Assuming means and stds are defined    
# Assuming `stacked_patches` is your numpy array of shape (3922, 3, 256, 256)
dataset = PatchesDataset(normalized_data)
batch_size = 16  # You can adjust the batch size as needed
shuffle = True  # Shuffling for training
data_loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

# Initialize your model (assuming no hyperparameter tuning is needed)
model = CROMA(patch_size=8, encoder_dim=768, encoder_layers=12, attention_heads=16, decoder_dim=512, decoder_layers=1, total_channels=3, num_patches=1024)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Example optimizer and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=6.777439597940801e-05)
criterion = torch.nn.MSELoss()  # Assuming MSE loss is appropriate for your task

# Variables to store loss values
epoch_losses = []

# Number of epochs
num_epochs = 100 
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
batch_size = 16  
num_patches = 1024 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mask_ratio =0.136

# Training loop
radar_mask_info=get_mask(batch_size, num_patches, device, mask_ratio)
for epoch in range(num_epochs):   
    model.train()  # Set model to training mode
    epoch_loss = 0.0
    for batch_idx, batch in enumerate(data_loader):
        print(batch.shape)
        inputs= batch  # No labels in SSL, inputs are used as targets        
        radar_mask_info=get_mask(batch_size, num_patches, device, mask_ratio)
        optimizer.zero_grad()
        outputs,pred_image = model(inputs,radar_mask_info)
        loss = criterion(outputs, inputs)  # Self-supervised: model tries to reproduce its input
        loss.backward()
        optimizer.step()        
        epoch_loss += loss.item() * inputs.size(0)
    epoch_loss /= len(data_loader.dataset)
    print(epoch_loss)
    epoch_losses.append(epoch_loss)
    scheduler.step()  # Update the learning rate
    
    

# Plot and save the loss curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), epoch_losses, marker='o', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('training_loss.png')
plt.close()
