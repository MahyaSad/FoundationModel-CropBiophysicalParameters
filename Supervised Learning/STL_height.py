import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import optuna
from torch.optim.lr_scheduler import CosineAnnealingLR
import optuna
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
import re
from sklearn.model_selection import train_test_split
from transformers import ViTModel
from einops import rearrange
from torch import nn, einsum
import math
import itertools
from torch import distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        radar_encodings = self.radar_encoder(imgs=radar_imgs, attn_bias=radar_masked_attn_bias)

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
        return radar_GAP


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
    
def exists(val):
    return val is not None

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
            # Adjust alibi to match attention_scores shape
            alibi = alibi[:, :, :attention_scores.shape[2], :attention_scores.shape[3]]
            attention_scores = attention_scores + alibi
        
        attn = attention_scores.softmax(dim=-1)        
        
        # Add specific print statements before einsum to debug shapes        
        out = einsum('b h i j, b h j d -> b h i d', attn, v)                
        out = rearrange(out, 'b h n d -> b n (h d)')
                
        out = self.out(out)                
        return out

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
        print('atten',attn.shape)
        print('V',v.shpae)
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


def load_datasets(folder_path):
    # Initialize lists to store the paths
    data_files = []
    label_files = []
    climate_files = []
    SLC_files = []
    S2_files = []

    # Iterate through each file in the folder
    for file in os.listdir(folder_path):
        if file.endswith('data.npy'):
            # Construct the full paths and add to the lists
            data_files.append(os.path.join(folder_path, file))
            
            corresponding_label_file = file.replace('data.npy', 'label.npy')
            label_files.append(os.path.join(folder_path, corresponding_label_file))
            
            corresponding_climate_file = file.replace('data.npy', 'climate.npy')
            climate_files.append(os.path.join(folder_path, corresponding_climate_file))
            
            corresponding_SLC_file = file.replace('data.npy', 'SLC.npy')
            SLC_files.append(os.path.join(folder_path, corresponding_SLC_file))
            
            corresponding_S2_file = file.replace('data.npy', 'S2_2.npy')
            S2_files.append(os.path.join(folder_path, corresponding_S2_file))

    # Sort files to ensure matching pairs are aligned
    data_files.sort()
    label_files.sort()
    climate_files.sort()
    SLC_files.sort()
    S2_files.sort()

    return data_files, label_files, climate_files, SLC_files, S2_files

def z_score_normalize1(data):
    # For samples with only one time step, compute the mean and std across the channel, width, and height
    mean = np.nanmean(data, axis=(1), keepdims=True) 
    std = np.nanstd(data, axis=(1), keepdims=True)  
    std[std == 0] = 1
    # Normalize the data
    normalized_data = (data - mean) / std
    return normalized_data
    
def z_score_normalize2(data):
    # For samples with only one time step, compute the mean and std across the channel, width, and height
    mean = np.nanmean(data[:,0,:,:,:], axis=(0,2,3), keepdims=True)  # Shape: [1,1, C, 1, 1]
    std = np.nanstd(data[:,0,:,:,:],axis=(0,2,3), keepdims=True)    # Shape: [1,1, C, 1, 1]
    # Avoid division by zero by setting std to 1 where std is zero
    std[std == 0] = 1
    # Normalize the data
    normalized_data=data
    normalized_data[:,0,:,:,:] = (data[:,0,:,:,:] - mean) / std
    return normalized_data

def pad_with_nans(data, target_channels):
    # Assumes data is in shape (batch, channels, height, width)
    current_channels = data.shape[1]
    if current_channels < target_channels:
        # Calculate how many channels need to be added
        needed_channels = target_channels - current_channels
        # Create a NaN array of the appropriate shape
        nan_padding = np.full((data.shape[0], needed_channels, data.shape[2], data.shape[3]), np.nan)
        # Concatenate along the channel dimension
        data = np.concatenate([data, nan_padding], axis=1)
    return data

def read_data_and_labels(data_files, label_files, climate_files, SLC_files, S2_files,N):
    all_data = []
    all_labels = []
    data_list=[]
    label_list=[]
    max_channels = 18
    data2016=[]
    N=N
    print(N)

    for data_path, label_path, climate_path, SLC_path, S2_path in zip(np.sort(data_files), np.sort(label_files), np.sort(climate_files), np.sort(SLC_files), np.sort(S2_files)):
        # Load each pair of data and label files
        # 2023 and 2022 starts with doy but 2016 starts wtih the features 
        patch_number = int(re.search(r'_patch_(\d+)_', data_path).group(1))
        data = np.load(data_path)
        label = np.load(label_path)
        SLC = np.load(SLC_path)
        new_S2 = np.load(S2_path)
        climate = np.load(climate_path)        
        if patch_number <= 6:
                    data2 = data[0, 1:6, 5:15]
                    expanded_label = label[0, 1:3, 5:15]
                    SLC2 = SLC[0, 1:, 2:]
                    new_S2 = new_S2[1:, 1:]
                    
                    climate = climate[0, 1:, 1:]
        elif patch_number <= 14 and patch_number > 6:
                    data2 = data[0, 1:6, 3:13]
                    expanded_label = label[0, 1:3, 3:13]
                    SLC2 = SLC[0, 1:, 3:]
                    new_S2 = new_S2[0, 1:, :-1]
                    climate = climate[0, 1:, :]
                    
                    
        elif patch_number == 16:
                    data2 = data[0, 1:6, 3:13]
                    expanded_label = label[0, 1:3, 3:13]
                    SLC2 = SLC[0, 1:, 3:]
                    new_S2 = new_S2[1:,:]
                    climate = climate[0, 1:, :]
                    
        else:
                    data2 = data[0, 1:6]
                    data2 = data2.reshape(-1, 1)
                    label2 = label[0, :, 0]
                    SLC2 = SLC[0, 1:, :]
                    SLC2 = SLC2.reshape(-1, 1)
                    new_S2 = new_S2[1:].reshape(-1, 1)
                    climate = climate.reshape(-1, 1)
                    
      
                    nan_columns = np.full((2, 9), np.nan)
                    
                    expanded_label = np.hstack((label2.reshape(-1, 1), nan_columns))

        combined_data = np.concatenate([data2[0:2,:],data2[4:5,:], new_S2, climate,data2[2:4],SLC2], axis=0)
       
        data_list.append(combined_data)
        label_list.append(expanded_label)
    data_array = np.concatenate(data_list, axis=1).transpose()
    label_array=np.concatenate(label_list, axis=1).transpose()
        
    scaler = StandardScaler()
    train_data = scaler.fit_transform(data_array)
    
    nan_columns = np.full((9, 14), np.nan)
    
    for k in range(N*10,train_data.shape[0]):

        expanded_data = np.vstack((train_data[k,:].reshape(1, -1), nan_columns))
        
        data2016.append(expanded_data)
        
    data_array2 = np.concatenate(data2016, axis=0)
    
    all_data=np.concatenate([train_data[:N*10,:], data_array2], axis=0)
    all_labels=label_array


    
    data_reshaped = all_data.reshape(all_data.shape[0], all_data.shape[1], 1, 1)

    # Extend the data from [time, channel, 1, 1] to [time, channel, 8, 8] by repeating
    data_extended = np.tile(data_reshaped, (1, 1, 8, 8))


    
    return data_extended,all_labels



class SARVWCDataset(Dataset):
    def __init__(self, all_data, all_labels):
        """
        all_data: A list of numpy arrays, each of shape (time_series_length, num_features)
        all_labels: A list of numpy arrays, each of shape (time_series_length)
        """
        self.all_data = [torch.tensor(data, dtype=torch.float32) for data in all_data]
        self.all_labels = [torch.tensor(labels, dtype=torch.float32) for labels in all_labels]

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return self.all_data[idx], self.all_labels[idx]

def remove_nan_samples(data):
   
    mask_2nd_dim = ~torch.isnan(data).any(dim=4).any(dim=3)

    # Select slices without NaNs in the second dimension
    result = data[mask_2nd_dim, :, :]  # Correct the dimension to apply the mask
    # Use the unsqueeze operations and assign it back to y if necessary

    if result.shape[0]>20:
        new_shape = (10, result.shape[0] // 10, result.shape[1], result.shape[2])
        result = result.reshape(new_shape)  # Reshape and assign the result back to 'result'
        filtered_data = result.unsqueeze(0)
    else: 
        filtered_data = result.unsqueeze(0).unsqueeze(0)
    return filtered_data
   
def remove_nan_samples_y(data):
    """
    Removes samples that contain any NaN values from the tensor.

    Args:
    data (torch.Tensor): Input data tensor with shape (batch_size, samples, features).

    Returns:
    torch.Tensor: Filtered data tensor without NaN samples, potentially with fewer samples.
    """
    # Check for NaN across the features dimension and reduce to a mask of shape (batch_size, samples)
    mask = ~torch.isnan(data).any(dim=2)


    # Apply the mask to filter samples. The mask is for the samples dimension.
    filtered_data = data[:, mask[0]]  # Apply mask for the first batch

    return filtered_data


        
class SupervisedModel(nn.Module):
    def __init__(self, pre_trained_vit_SAR,pre_trained_vit_Opt, input_dim, hidden_dim, output_dim_vwc, output_dim_height, dropout):
        super().__init__()
        self.pre_trained_vit_SAR = pre_trained_vit_SAR.to(device)
        self.pre_trained_vit_Opt = pre_trained_vit_Opt.to(device)
        
        for param in self.pre_trained_vit_SAR.parameters():
            param.requires_grad = False
            
        for param in self.pre_trained_vit_Opt.parameters():
            param.requires_grad = False
        
        self.hidden_dim = hidden_dim
        self.output_dim_vwc = output_dim_vwc
        self.output_dim_height = output_dim_height
        self.input_dim = input_dim
        self.dropout = dropout
        
        self.output_projection_vwc = nn.Linear(hidden_dim, output_dim_vwc)
        self.output_projection_height = nn.Linear(hidden_dim, output_dim_height)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.conv1d = nn.Conv1d(12, hidden_dim, kernel_size=3, padding=1)  # Assume input_dim is the depth of feature channels




    def forward(self, x):
        clean_x = remove_nan_samples(x)
        batch_size, seq_len, C, H, W = clean_x.shape
        gru_inputs = torch.zeros(batch_size, seq_len, self.input_dim*2).to(clean_x.device)
        
        radar_mask_info = get_mask(batch_size, 1024, device, 0.136)
   
        for t in range(seq_len):
            radar_GAP = self.pre_trained_vit_SAR(clean_x[:, t, :3, :, :], radar_mask_info)
            
            Opt_GAP = self.pre_trained_vit_Opt(clean_x[:, t, 3:6, :, :], radar_mask_info)
          
            fused_features = torch.cat((radar_GAP, Opt_GAP), dim=1)

            gru_inputs[:, t, :] = fused_features
            
        output_projection = nn.Linear(768*2, 6)
        XX=output_projection(gru_inputs)
            
        additional_features = clean_x[:, :, :, :, :].mean(dim=(-1, -2))
        
        features = torch.cat((additional_features[:,:,0:1],additional_features[:,:,4:6],additional_features[:,:,6:9]), axis=-1)
        
        x = torch.cat((XX, features), dim=-1)
        
        # Change input shape to (batch_size, channels, seq_len) for Conv1D
        x = x.permute(0, 2, 1)

        # Conv1D and ReLU activation
        x = self.conv1d(x)
        x = self.relu(x)

        # Apply dropout
        x = self.dropout_layer(x)


        x = x.permute(0, 2, 1)


        # Output layers
        output_vwc = self.output_projection_vwc(x)
        #print(output_vwc.shape)
        
        
        return output_vwc

# Define your supervised model
# input_dim: The embedding dimension output by your pre-trained ViT
# hidden_dim, num_layers: Dimensions for the transformer model
# output_dim: The output dimension which should match your label dimension, e.g., 1 for VWC
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
def create_sequences(data, labels, sequence_length=10):
    sequences = []
    sequence_labels = []

    for i in range(0, len(data) - sequence_length + 1, sequence_length):
        sequences.append(data[i:i+sequence_length])
        sequence_labels.append(labels[i:i+sequence_length])

    return np.array(sequences), np.array(sequence_labels)


# Define your folder paths
train_val_folder_path = 'soy_nonirrigated'
test_folder_path = 'soy_nonirrigated_test'

# Load datasets
data_files, label_files, climate_files, SLC_files, S2_files = load_datasets(train_val_folder_path)
test_data_files, test_label_files, test_climate_files, test_SLC_files, test_S2_files = load_datasets(test_folder_path)


# Read data and labels
all_data, all_labels = read_data_and_labels(data_files, label_files, climate_files, SLC_files, S2_files,N=8)
test_data, test_labels = read_data_and_labels(test_data_files, test_label_files, test_climate_files, test_SLC_files, test_S2_files,N=5)

# Initialize scalers
vwc_scaler = MinMaxScaler()
height_scaler = MinMaxScaler()
leaves_scaler = MinMaxScaler()

# Fit and transform the labels
all_labels[:, 0] = vwc_scaler.fit_transform(all_labels[:, 0].reshape(-1, 1)).flatten()
all_labels[:, 1] = height_scaler.fit_transform(all_labels[:, 1].reshape(-1, 1)).flatten()


# Apply the same transformation to test labels
test_labels[:, 1] = height_scaler.transform(test_labels[:, 1].reshape(-1, 1)).flatten()



# Initialize your model (assuming no hyperparameter tuning is needed)
pre_trained_vit_SAR = CROMA(patch_size=8, encoder_dim=768, encoder_layers=12, attention_heads=16, decoder_dim=512, decoder_layers=1, total_channels=3, num_patches=1024)
pre_trained_vit_Opt = CROMA(patch_size=8, encoder_dim=768, encoder_layers=12, attention_heads=16, decoder_dim=512, decoder_layers=1, total_channels=3, num_patches=1024)

# Load the state_dict from the file
state_dict_SAR = torch.load('model_path_50_SAR.pth')
state_dict_Opt = torch.load('model_path_50_Opt.pth')

# Filter out decoder keys
encoder_state_dict_SAR = {k: v for k, v in state_dict_SAR.items() if not k.startswith('decoder.')}
encoder_state_dict_Opt = {k: v for k, v in state_dict_Opt.items() if not k.startswith('decoder.')}

# Load the filtered state_dict
pre_trained_vit_SAR.load_state_dict(encoder_state_dict_SAR, strict=False)
pre_trained_vit_Opt.load_state_dict(encoder_state_dict_Opt, strict=False)

pre_trained_vit_SAR.eval()
pre_trained_vit_Opt.eval()

# Create sequences
sequences, sequence_labels = create_sequences(all_data, all_labels, sequence_length=10)
test_sequences, test_sequence_labels = create_sequences(test_data, test_labels, sequence_length=10)



# Split the sequences and their labels into training and validation sets
train_sequences, val_sequences, train_sequence_labels, val_sequence_labels = train_test_split(
    sequences, sequence_labels, test_size=0.1, random_state=42)

def calculate_metrics(true_values, predicted_values, scaler):
    true_values_original = scaler.inverse_transform(true_values.reshape(-1, 1)).flatten()
    predicted_values_original = scaler.inverse_transform(predicted_values.reshape(-1, 1)).flatten()
    
    print(predicted_values_original)
    
    mae = np.mean(np.abs(predicted_values_original - true_values_original))
    mse = mean_squared_error(true_values_original, predicted_values_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values_original, predicted_values_original)
    
    return mae, rmse, r2


lr = 0.00024369879759874696
batch_size = 8
num_epochs = 4000
patience = 13
hidden_dim = 1024
dropout = 0.13627231264529863
weight_decay = 2.950811865425501e-05
lr_cosine_init = 0.00999416680763606
lr_cosine_cycles = 4



all_data_tensor = torch.tensor(train_sequences, dtype=torch.float32)
all_labels_tensor = torch.tensor(train_sequence_labels, dtype=torch.float32)
full_dataset = TensorDataset(all_data_tensor, all_labels_tensor)

# Prepare test dataset
test_data_tensor = torch.tensor(test_sequences, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_sequence_labels, dtype=torch.float32)
test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize KFold
best_model = SupervisedModel(pre_trained_vit_SAR, pre_trained_vit_Opt, input_dim=768, hidden_dim=hidden_dim, output_dim_vwc=1, output_dim_height=1, dropout=dropout).to(device)
best_model.load_state_dict(torch.load('best_model_fold_8.pth'))
best_model.eval()
test_predictions_vwc, test_actuals_vwc = [], []

with torch.no_grad():
    for X_test, y_test in test_loader:
        X_test, y_test = X_test.to(device), y_test.to(device)
        for i in range(y_test.size(0)):
            Y_test = remove_nan_samples_y(y_test[i:i+1, :, :])
            y_vwc_test = Y_test[:, :, 1].to(device)
            output_vwc_test = best_model(X_test[i:i+1, :, :, :, :])
            test_predictions_vwc.extend(output_vwc_test[:, :, 0].cpu().numpy().flatten())
            test_actuals_vwc.extend(y_vwc_test.cpu().numpy().flatten())
            


# Calculate metrics
mae_vwc, rmse_vwc, r2_vwc = calculate_metrics(np.array(test_actuals_vwc), np.array(test_predictions_vwc), height_scaler)

print(f"Test Results:")
print(f"VWC - MAE: {mae_vwc:.4f}, RMSE: {rmse_vwc:.4f}, R^2: {r2_vwc:.4f}")

