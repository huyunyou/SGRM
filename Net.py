from utils import *
from ultralytics.nn.modules.conv import Conv
import torch.nn.init as init

class ConvBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False)
        self.relu1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False)
        self.relu2 = nn.LeakyReLU(inplace=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                # m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return x


class ADown(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)  # 3x3 conv with stride 2
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)  # 1x1 conv

    def forward(self, x):
        x = F.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)  # Path 1: Conv + AvgPool
        x2 = F.max_pool2d(x2, 3, 2, 1)  # Path 2: MaxPool
        x2 = self.cv2(x2)  # Path 2: Conv
        return torch.cat((x1, x2), 1)


class AttentionGate(nn.Module):
    def __init__(self, g_channels, x_channels, out_channels):
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            # nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class ADGAUNet(nn.Module):
    def __init__(self, in_ch=4, base_ch=32, out_ch=4):
        super().__init__()

        # ===== Encoder =====
        self.enc1 = ConvBlock(in_ch, base_ch, base_ch)
        # self.pool1 = nn.AvgPool2d(2)
        self.pool1_max = nn.MaxPool2d(2)
        self.adown1 = ADown(base_ch, base_ch)
        # self.pcdm1 = PCDM(base_ch, base_ch)

        self.enc2 = ConvBlock(base_ch, base_ch * 2, base_ch * 2)
        # self.pool2 = nn.AvgPool2d(2)
        self.pool2_max = nn.MaxPool2d(2)
        # self.pcdm2 = PCDM(base_ch*2,base_ch*2)
        self.adown2 = ADown(base_ch * 2, base_ch * 2)

        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, base_ch * 4)

        # ===== Decoder =====
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = ConvBlock(base_ch * 6, base_ch * 4, base_ch * 2)
        self.Att1 = AttentionGate(base_ch * 4, base_ch * 2, base_ch * 2)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = ConvBlock(base_ch * 3, base_ch * 2, base_ch)
        self.Att2 = AttentionGate(base_ch * 2, base_ch, base_ch * 2)

        self.out = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        # ===== Encoder =====
        f1 = self.enc1(x)
        # x1 = self.pool1(f1)
        # x1_1 = self.pool1_max(f1)
        x1_1 = self.adown1(f1)

        f2 = self.enc2(x1_1)
        # f2 = self.enc2(torch.cat([x1, x1_1], dim=1))
        # x2 = self.pool2(f2)
        # x2_1 = self.pool2_max(f2)
        x2_1 = self.adown2(f2)

        f3 = self.enc3(x2_1)
        # f3 = self.enc3(torch.cat([x2, x2_1], dim=1))

        # ===== Decoder =====
        x4 = self.up1(f3)
        x4_1 = self.Att1(x4, f2)
        x5 = torch.cat([x4, x4_1], dim=1)
        x6 = self.dec1(x5)

        x8 = self.up2(x6)
        x8_1 = self.Att2(x8, f1)
        x9 = torch.cat([x8, x8_1], dim=1)
        x10 = self.dec2(x9)

        return self.out(x10)