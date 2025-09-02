import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from mamba_ssm import Mamba
from utils import FDS
from torchvision.models import resnet18

class MambaModel(nn.Module):
    def __init__(self, d_model, max_length=30):
        super(MambaModel, self).__init__()
        self.linear = nn.Linear(in_features=21, out_features=d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_length)
        self.mamba = Mamba(d_model=d_model, d_state=32, expand=4)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor):
        x = self.pos_encoder(self.linear(x))
        y = self.mamba(x)
        y_flip = self.mamba(x.flip([-2])).flip([-2])
        y = torch.cat((y, y_flip), dim=-1)
        y = self.global_pool(y.permute(0, 2, 1)).squeeze(-1)
        return y


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout_rate=0.1):
        super(MLP, self).__init__()
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim] * num_layers
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        for i in range(len(hidden_dim) - 1):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        layers.append(nn.Linear(hidden_dim[-1], output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-torch.log(torch.FloatTensor([10000.0])) / d_model))  # (d_model/2,)
        pe[:, 0::2] = torch.sin(position * div_term)  # even dimension
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dimension
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        x: (B, N, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class MHAModel(nn.Module):
    def __init__(self, d_model, max_length=50):
        super(MHAModel, self).__init__()
        self.linear = nn.Linear(in_features=21, out_features=d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_length)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor):
        x = self.pos_encoder(self.linear(x))  # [batch, seq_len, d_model]
        
        y, _ = self.self_attn(x, x, x)
        
        x_flip = x.flip([-2])
        y_flip, _ = self.self_attn(x_flip, x_flip, x_flip)
        y_flip = y_flip.flip([-2])
        
        y = torch.cat((y, y_flip), dim=-1)  # [batch, seq_len, 2*d_model]
        
        y = self.global_pool(y.permute(0, 2, 1))  # [batch, 2*d_model, 1]
        return y.squeeze(-1)  # [batch, 2*d_model]  
    

class DMutaPeptide(nn.Module):
    def __init__(self, q_encoder='lstm', classes=1, channels=128, dir=False, gf=False, fusion='mlp', non_siamese=False):
        super().__init__()
        self.classes = classes
        self.DIR = dir
        self.gf = gf
        self.fusion_method = fusion  # 融合方式
        self.non_siamese = non_siamese
        # 拼接后维度设定为 channels * 4
        final_dim = channels * 4


        # 初始化编码器
        if q_encoder == 'lstm':
            self.q_encoder = nn.LSTM(
                input_size=21,
                hidden_size=channels,
                num_layers=2,
                batch_first=True,
                dropout=0.1,
                bidirectional=True
            )
        elif q_encoder == 'gru':
            self.q_encoder = nn.GRU(
                input_size=21,
                hidden_size=channels,
                num_layers=2,
                batch_first=True,
                dropout=0.1,
                bidirectional=True
            )
        elif q_encoder == 'mamba':
            self.q_encoder = MambaModel(channels, 30)
        elif q_encoder == 'mha':
            self.q_encoder = MHAModel(channels, 30)
        else:
            raise NotImplementedError
        
        if non_siamese:
            self.q_encoder_2 = deepcopy(self.q_encoder)
        else:
            self.q_encoder_2 = self.q_encoder
        
        if self.fusion_method == 'diff':
            final_dim //= 2

        if gf:
            self.g_encoder = MLP(1024, [512, 256, 128], channels * 2, dropout_rate=0.3)
            final_dim += channels * 2

        if self.fusion_method == 'att':
            embed_dim = channels * 2
            self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4 if gf else 2, batch_first=True)
            
        if self.DIR:
            self.FDS = FDS(final_dim)

        self.fc = nn.Sequential(
            nn.Linear(final_dim, 128),
            nn.Mish(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.Mish(),
            nn.Dropout(0.3),
            nn.Linear(64, self.classes)
        )

    def norm(self, x, dim=-1, p=2):
        return F.normalize(x, p=p, dim=dim)

    def forward(self, x, labels=None, epoch=0):
        if self.gf:
            seq1, seq2, gf = x
        else:
            seq1, seq2 = x
        fusion = []

        if self.q_encoder.__class__.__name__ in ['LSTM', 'GRU']:
            fusion.append(self.norm(self.q_encoder(seq1)[0][:, -1, :]))
            fusion.append(self.norm(self.q_encoder_2(seq2)[0][:, -1, :]))
        else:
            fusion.append(self.norm(self.q_encoder(seq1)))
            fusion.append(self.norm(self.q_encoder_2(seq2)))
        
        if self.gf:
            fusion.append(self.g_encoder(gf))

        if self.fusion_method == 'mlp':
            fusion = torch.cat(fusion, dim=-1)
        elif self.fusion_method == 'diff':
            fusion = torch.cat([fusion[1] - fusion[0]] + fusion[2:], dim=-1)
        elif self.fusion_method == 'att':
            tokens = torch.stack(fusion, dim=1)  # embed_dim 应该为 final_dim//2
            attn_output, _ = self.attn(tokens, tokens, tokens)
            fusion = attn_output.reshape(attn_output.size(0), -1)
        else:
            raise ValueError("Invalid fusion method: choose either 'mse' or 'att'.")

        if self.DIR:
            features = fusion
            fusion = self.FDS.smooth(fusion, labels, epoch)
        
        pred = self.fc(fusion).squeeze(-1)

        if self.DIR:
            return pred, features
        else:
            return pred


class CNNEncoder(nn.Module):
    def __init__(self, feature_dim=256, base_channels=16, in_dim=3):
        super(CNNEncoder, self).__init__()
        
        # 卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.Mish(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.Mish(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.Mish(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(base_channels * 4, feature_dim)
        
    def forward(self, img):
        fused_conv = self.conv(img)
        pooled = self.adaptive_pool(fused_conv)  # [B, base_channels*4, 1, 1]

        flattened = pooled.view(pooled.size(0), -1)  # [B, base_channels*4]
        feature_vector = self.fc(flattened)          # [B, feature_dim]
        return feature_vector


class DMutaPeptideCNN(nn.Module):
    def __init__(self, q_encoder='cnn', classes=1, channels=16, dir=False, gf=False, side_enc=None, fusion='mlp', non_siamese=False):
        super().__init__()
        self.classes = classes
        self.DIR = dir
        self.gf = gf
        self.fusion_method = fusion
        self.non_siamese = non_siamese
        vector_dim = 512
        final_dim = vector_dim * 2

        if q_encoder == 'cnn':
            self.q_encoder = CNNEncoder(feature_dim=vector_dim, base_channels=channels)
        elif q_encoder == 'rn18':
            self.q_encoder = resnet18_backbone(pretrained=True)
        if non_siamese:
            self.q_encoder_2 = deepcopy(self.q_encoder)
        else:
            self.q_encoder_2 = self.q_encoder

        if side_enc:
            self.side_enc = True
            if side_enc == 'lstm':
                self.side_encoder = nn.LSTM(
                    input_size=21,
                    hidden_size=256,
                    num_layers=2,
                    batch_first=True,
                    dropout=0.1,
                    bidirectional=True
                )
            elif side_enc == 'mamba':
                self.side_encoder = MambaModel(256, 30)
            else:
                raise NotImplementedError
            
            final_dim += vector_dim * 2

            if non_siamese:
                self.side_encoder_2 = deepcopy(self.side_encoder)
            else:
                self.side_encoder_2 = self.side_encoder
        else:
            self.side_enc = False
        
        if self.fusion_method == 'diff':
            final_dim //= 2
        
        if gf:
            self.g_encoder = MLP(1024, [512, 256, 128], vector_dim, dropout_rate=0.3)
            final_dim += vector_dim

        if self.fusion_method == 'att':
            embed_dim = vector_dim
            self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4 if gf else 2, batch_first=True)
  
        if self.DIR:
            self.FDS = FDS(final_dim)

        self.fc = nn.Sequential(
            nn.Linear(final_dim, 128),
            nn.Mish(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.Mish(),
            nn.Dropout(0.3),
            nn.Linear(64, self.classes)
        )

    def norm(self, x, dim=-1, p=2):
        return F.normalize(x, p=p, dim=dim)

    def forward(self, x, labels=None, epoch=0):
        if self.gf:
            seq1, seq2, gf = x
        else:
            seq1, seq2 = x

        if self.side_enc:
            seq1_seq = seq1[1]
            seq1 = seq1[0]
            seq2_seq = seq2[1]
            seq2 = seq2[0]

        fusion = []

        fusion.append(self.norm(self.q_encoder(seq1)))
        fusion.append(self.norm(self.q_encoder_2(seq2)))
        if self.side_enc:
            if self.side_encoder.__class__.__name__ == 'MambaModel':
                fusion.append(self.norm(self.side_encoder(seq1_seq)))
                fusion.append(self.norm(self.side_encoder_2(seq2_seq)))
            else:
                fusion.append(self.norm(self.side_encoder(seq1_seq)[0][:, -1, :]))
                fusion.append(self.norm(self.side_encoder_2(seq2_seq)[0][:, -1, :]))
        
        if self.gf:
            fusion.append(self.g_encoder(gf))

        if self.fusion_method == 'mlp':
            fusion = torch.cat(fusion, dim=-1)
        elif self.fusion_method == 'diff':
            if not self.side_enc:
                fusion = torch.cat([fusion[1] - fusion[0]] + fusion[2:], dim=-1)
            else:
                fusion = torch.cat([fusion[1] - fusion[0], fusion[3] - fusion[2]] + fusion[4:], dim=-1)
        elif self.fusion_method == 'att':
            tokens = torch.stack(fusion, dim=1)
            attn_output, _ = self.attn(tokens, tokens, tokens)
            fusion = attn_output.reshape(attn_output.size(0), -1)
        else:
            raise ValueError("Invalid fusion method: choose either 'mse' or 'att'.")

        if self.DIR:
            features = fusion
            fusion = self.FDS.smooth(fusion, labels, epoch)
        
        pred = self.fc(fusion).squeeze(-1)

        if self.DIR:
            return pred, features
        else:
            return pred
        

def resnet18_backbone(pretrained=False):
    weights = None
    if pretrained:
        weights = 'IMAGENET1K_V1'
    model = resnet18(weights=weights, progress=False)
    return torch.nn.Sequential(*list(model.children())[:-1], nn.Flatten())