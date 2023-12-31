import numpy as np
import torch
from torch import nn

from GradientReversalLayer import GradientReversalLayer
from PositionalEncoding import PositionalEncoding


# noinspection DuplicatedCode
class RTGRFID(nn.Module):
    def __init__(self, seq_len: int, num_gestures: int, num_users: int) -> None:
        super().__init__()
        self.num_gestures = num_gestures
        self.num_users = num_users
        self.seq_len = seq_len

        embed_dim = 2 * 3 * 3
        self.pos_encoder_array = PositionalEncoding(embed_dim, dropout=0)
        self.pos_encoder_cat = PositionalEncoding(embed_dim * 2, dropout=0)

        self.feature_extractors_array_1 = nn.MultiheadAttention(embed_dim, 1, batch_first=True)
        self.feature_extractors_array_2 = nn.MultiheadAttention(embed_dim, 1, batch_first=True)

        self.gesture_labeler = nn.MultiheadAttention(embed_dim * 2, 1, batch_first=True)
        self.gesture_label_extractors = [nn.Linear(embed_dim * 2, 1) for _ in range(self.seq_len)]
        self.gesture_mlp = nn.Sequential(
            nn.Linear(seq_len, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_gestures)
        )

        self.user_labeler = nn.MultiheadAttention(embed_dim * 2, 1, batch_first=True)
        self.user_label_extractors = [nn.Linear(embed_dim * 2, 1) for _ in range(self.seq_len)]
        self.user_mlp = nn.Sequential(
            nn.Linear(seq_len, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, self.num_users),
        )

    def forward(self, x, alpha):
        seq_1: list[torch.Tensor] = []
        seq_2: list[torch.Tensor] = []
        for frame in x:
            # 每个frame包含两个阵列
            rss_tensor, phase_tensor = frame[0]
            merged_array_feature = torch.stack(
                (rss_tensor, phase_tensor), 0)
            seq_1.append(merged_array_feature.flatten())

            # 每个frame包含两个阵列
            rss_tensor, phase_tensor = frame[1]
            merged_array_feature = torch.cat(
                (rss_tensor, phase_tensor), 1)
            seq_2.append(merged_array_feature.flatten())

        seq_1: torch.Tensor = torch.stack(seq_1)
        # seq_1: torch.Tensor = self.pos_encoder_array(seq_1)
        seq_2: torch.Tensor = torch.stack(seq_2)
        # seq_2: torch.Tensor = self.pos_encoder_array(seq_2)

        attn_output_1_g, _ = self.feature_extractors_array_1(seq_1, seq_1, seq_1)
        attn_output_2_g, _ = self.feature_extractors_array_2(seq_2, seq_2, seq_2)
        features_seq: torch.Tensor = torch.cat((attn_output_1_g, attn_output_2_g), dim=1)
        # features_seq = self.pos_encoder_cat(features_seq)

        # Gestures Labeler
        attn_output_g, _ = self.gesture_labeler(features_seq, features_seq, features_seq)
        mlp_g_input = torch.tensor([])
        for (index, item) in enumerate(attn_output_g):
            label = self.gesture_label_extractors[index](item)
            mlp_g_input = torch.cat((mlp_g_input, label), 0)
        gestures_output = self.gesture_mlp(mlp_g_input)

        # User Labeler
        features_seq_reversed = GradientReversalLayer.apply(features_seq, alpha)
        attn_output_u, _ = self.user_labeler(features_seq_reversed, features_seq_reversed, features_seq_reversed)
        mlp_u_input = torch.tensor([])
        for (index, item) in enumerate(attn_output_u):
            label = self.user_label_extractors[index](item)
            mlp_u_input = torch.cat((mlp_u_input, label), 0)
        user_output = self.user_mlp(mlp_u_input)

        return gestures_output, user_output
