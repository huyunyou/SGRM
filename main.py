import os
from imageio import imread, imsave
import time
from pytorch_wavelets import DWTForward, DWTInverse
from loss import *
from Net import ADGAUNet

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def single_level_dwt(img_tensor):
    dwt = DWTForward(J=1, wave="haar", mode="zero")

    with torch.no_grad():
        Yl, Yh = dwt(img_tensor)

    LL = Yl
    LH = Yh[0][:, :, 0, :, :]
    HL = Yh[0][:, :, 1, :, :]
    HH = Yh[0][:, :, 2, :, :]

    return LL, LH, HL, HH


class SGRM(nn.Module):
    def __init__(
        self,
        noise,
        sigma_min_L=0.95,
        sigma_max_L=1.00,
        sigma_min_H=0.75,       # 0.75-0.80  0.65-0.70
        sigma_max_H=0.80,
        num_masks=50,
        sparse_ratio = 0.55
    ):
        super().__init__()
        self.sigma_min_L = sigma_min_L
        self.sigma_max_L = sigma_max_L
        self.sigma_min_H = sigma_min_H
        self.sigma_max_H = sigma_max_H
        self.num_masks = num_masks
        self.sparse_ratio = sparse_ratio
        self.num_masks_per_iter = 1

        self.register_buffer("mask_bank1", None)
        self.register_buffer("mask_bank2", None)
        self.register_buffer("mask_bank3", None)
        self.register_buffer("mask_bank4", None)
        self.net = ADGAUNet()
        self.gaussian_mask(noise)

    def generate_sparse_mask_prob(self,shape, sparsity_ratio,device):
        mask = torch.rand(size=shape,device=device) < sparsity_ratio
        return mask.float()

    def gaussian_mask(self, noise):
        masks = []
        for i in range(self.num_masks + 1):
            std_m = torch.empty(1).uniform_(self.sigma_min_L, self.sigma_max_L).item()
            # binary_mask = torch.bernoulli(torch.full(noise10.shape, self.K))
            m = torch.normal(mean=0, std=std_m, size=noise.shape)
            # ss = torch.rand(size=noise10.shape) < self.sparse_ratio
            # ss = ss.float()
            # temp = ss*m + (1-ss)
            temp = m
            # m = torch.where(binary_mask == 1, torch.tensor(1), normal_mask)
            temp = temp.squeeze(0)
            masks.append(temp)
            self.mask_bank1 = torch.stack(masks, dim=0)

        masks = []
        for i in range(self.num_masks + 1):
            std_m = torch.empty(1).uniform_(self.sigma_min_H, self.sigma_max_H).item()
            m = torch.normal(mean=0, std=std_m, size=noise.shape)
            # ss = torch.rand(size=noise10.shape) < self.sparse_ratio
            # ss = ss.float()
            # temp = ss * m + (1 - ss)
            temp = m
            temp = temp.squeeze(0)
            masks.append(temp)
            self.mask_bank2 = torch.stack(masks, dim=0)

        masks = []
        for i in range(self.num_masks + 1):
            std_m = torch.empty(1).uniform_(self.sigma_min_H, self.sigma_max_H).item()
            m = torch.normal(mean=0, std=std_m, size=noise.shape)
            # ss = torch.rand(size=noise10.shape) < self.sparse_ratio
            # ss = ss.float()
            # temp = ss * m + (1 - ss)
            temp = m
            temp = temp.squeeze(0)
            masks.append(temp)
            self.mask_bank3 = torch.stack(masks, dim=0)

        masks = []
        for i in range(self.num_masks + 1):
            std_m = torch.empty(1).uniform_(self.sigma_min_H, self.sigma_max_H).item()
            m = torch.normal(mean=0, std=std_m, size=noise.shape)
            # ss = torch.rand(size=noise10.shape) < self.sparse_ratio
            # ss = ss.float()
            # temp = ss * m + (1 - ss)
            temp = m
            temp = temp.squeeze(0)
            masks.append(temp)
            self.mask_bank4 = torch.stack(masks, dim=0)

    def select_masks(self, index=1):
        if index == 1:
            idx = torch.randint(
                0,
                self.mask_bank1.shape[0],
                (self.num_masks_per_iter,),
                device=self.mask_bank1.device,
            )
            return self.mask_bank1[idx]
        if index == 2:
            idx = torch.randint(
                0,
                self.mask_bank2.shape[0],
                (self.num_masks_per_iter,),
                device=self.mask_bank2.device,
            )
            return self.mask_bank2[idx]
        if index == 3:
            idx = torch.randint(
                0,
                self.mask_bank3.shape[0],
                (self.num_masks_per_iter,),
                device=self.mask_bank3.device,
            )
            return self.mask_bank3[idx]
        if index == 4:
            idx = torch.randint(
                0,
                self.mask_bank4.shape[0],
                (self.num_masks_per_iter,),
                device=self.mask_bank4.device,
            )
            return self.mask_bank4[idx]

    def forward(self, y,kk=-1):
        mask1 = self.select_masks(1)  # [K, C, H, W]
        mask2 = self.select_masks(2)  # [K, C, H, W]
        mask3 = self.select_masks(3)  # [K, C, H, W]
        mask4 = self.select_masks(4)  # [K, C, H, W]
        mask_ch4 = torch.cat([mask1, mask2, mask3, mask4], dim=1)
        # if kk>=1000:
        #     ss = self.generate_sparse_mask_prob(mask_ch4.shape,1-0.1*math.floor(kk/500.0),mask_ch4.device)
        #     temp = ss * mask_ch4 + (1 - ss)
        # else:
        #     temp = mask_ch4
        ss = self.generate_sparse_mask_prob(mask_ch4.shape,0.55,mask_ch4.device)
        temp = ss * mask_ch4 + (1 - ss)
        ym = y * temp  # [K, C, H, W]
        preds = self.net(ym)  # [K, C, H, W]
        return preds

def train_sgrm(model, noisy_img, epochs=1000, lr=1e-3, device="cuda"):

    model = model.to(device)
    noisy_img = noisy_img.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.train()

    pea = PEALoss()
    pea = pea.to(device)

    for i in range(epochs + 1):
        optimizer.zero_grad()

        preds = model(noisy_img,i)
        # loss = F.mse_loss(preds, noisy_img)
        loss = pea(preds, noisy_img)

        loss.backward()
        optimizer.step()
        scheduler.step()

    return model


@torch.no_grad()
def inference(model, noisy_img, device="cuda"):
    model = model.to(device)
    noisy_img = noisy_img.to(device)

    model.eval()
    # pred = model.net(noisy_img)
    # pred = pred.cpu()

    # result = model(noisy_img)
    # for i in range(99):
    #     pred = model(noisy_img)
    #     result = torch.maximum(result, pred)

    outputs = []
    for i in range(99):
        pred = model(noisy_img,-1)
        outputs.append(pred)
    stacked = torch.stack(outputs, dim=0)
    result = torch.mean(stacked, dim=0)
    pred = result.cpu()

    LL_P = pred[:, 0:1, :, :]
    LH_P = pred[:, 1:2, :, :]
    HL_P = pred[:, 2:3, :, :]
    HH_P = pred[:, 3:4, :, :]

    idwt = DWTInverse(wave="haar", mode="zero")
    Yh_recombined = (torch.stack([LH_P, HL_P, HH_P], dim=2),)
    x_recon = idwt((LL_P, Yh_recombined))

    return x_recon


def main(img_path, img_path_GT, save_path):

    I, Q, U, DOP, DOP_r = get_stokes_vector_image(img_path)

    # _, _, _, _, DOP_gt = get_stokes_vector_image(img_path_GT)
    DOP_gt = get_single_dolp_image(img_path_GT)

    LL_P, LH_P, HL_P, HH_P = single_level_dwt(DOP)
    noisy = [LL_P, LH_P, HL_P, HH_P]

    noisy_img = torch.cat(noisy, dim=1)

    start_time = time.time()

    model = SGRM(LL_P)

    print(
        "The number of parameters of the network is: ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    
    model = train_sgrm(model, noisy_img, epochs=3000)
    out = inference(model, noisy_img)

    elapsed = time.time() - start_time

    out1 = out.cpu().squeeze().numpy()
    out1 = (out1 - np.min(out1)) / (np.max(out1) - np.min(out1))
    imsave(save_path, (out1 * 255).astype("uint8"))


if __name__ == "__main__":
    img_path = "./data/RealWorld/noise"
    img_path_GT = "./data/RealWorld/GT"
    out_path = "./output/RealWorld/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    image_files = sorted(os.listdir(img_path))
    for image_file in image_files:
        image_path = os.path.join(img_path, image_file)
        image_gt_path = os.path.join(img_path_GT, image_file)
        main(image_path, image_gt_path, out_path + image_file + ".bmp")
