import numpy as np
import torch
from torch import nn


class RTGRFID(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor_phase = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 3)
        )
        self.feature_extractor_rssi = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 3)
        )

    def forward(self, x):
        for frame in x:
            merged_feature = torch.tensor([])
            for array in frame:
                rss, phase = array
                rss, phase = np.array([rss]), np.array([phase])
                rss_tensor = torch.from_numpy(rss)
                phase_tensor = torch.from_numpy(phase)
                c_rss = self.feature_extractor_rssi(rss_tensor)[0]
                c_phase = self.feature_extractor_phase(phase_tensor)[0]
                merged_array_feature = torch.cat(
                    (c_rss, c_phase), 1)
                merged_feature = torch.cat((merged_feature, merged_array_feature), 0)
            flattened_feature = merged_feature.flatten()
            print(flattened_feature)
