import numpy as np
import torch
from torch import nn


class RTGRFID(nn.Module):
    def __init__(self, seq_len: int, num_classes: int) -> None:
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        embed_dim = 4 * 3 * 3
        self.labeler = nn.MultiheadAttention(embed_dim, 1, batch_first=True)
        self.label_extractors = [nn.Linear(embed_dim, 1) for _ in range(seq_len)]
        self.mlp = nn.Sequential(
            nn.Linear(seq_len, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            # nn.Softmax(dim=0)
        )

    def forward(self, x):
        seq: list[torch.Tensor] = []
        for frame in x:
            merged_feature = torch.tensor([])
            # 每个frame包含两个阵列
            for array in frame:
                rss, phase = array
                rss, phase = np.array([rss]), np.array([phase])
                rss_tensor = torch.from_numpy(rss)
                phase_tensor = torch.from_numpy(phase)
                # c_rss = self.feature_extractor_rssi(rss_tensor)[0]
                # c_phase = self.feature_extractor_phase(phase_tensor)[0]
                c_rss = rss_tensor
                c_phase = phase_tensor
                merged_array_feature = torch.cat(
                    (c_rss, c_phase), 1)
                merged_feature = torch.cat((merged_feature, merged_array_feature), 0)
            flattened_feature = merged_feature.flatten()
            seq.append(flattened_feature)
        seq: torch.Tensor = torch.stack(seq)
        query = seq
        key = seq
        value = seq
        attn_output, attn_output_weights = self.labeler(query, key, value)
        mlp_input = torch.tensor([])
        for (index, item) in enumerate(attn_output):
            label = self.label_extractors[index](item)
            mlp_input = torch.cat((mlp_input, label), 0)
        return self.mlp(mlp_input)
