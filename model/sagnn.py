import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import time
import cv2
from model.mpnn import MPNN
from model.mpnn import MPNN2
from model.mpnn import MPNN3
import model.resnet as models


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


def get_vgg16_layer(model):
    layer0_idx = range(0, 7)
    layer1_idx = range(7, 14)
    layer2_idx = range(14, 24)
    layer3_idx = range(24, 34)
    layer4_idx = range(34, 43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0, layer1, layer2, layer3, layer4


class SAGNN(nn.Module):
    def __init__(self, layers=50, classes=2, zoom_factor=8,
                 criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d,
                 pretrained=True, sync_bn=True, shot=1, ppm_scales=[60, 15, 8], no_pass=1, device_number=3):
        super(SAGNN, self).__init__()
        assert layers in [50, 101, 152]
        print(ppm_scales)
        assert classes > 1
        from torch.nn import BatchNorm2d as BatchNorm
        self.zoom_factor = zoom_factor
        self.criterion = criterion
        self.shot = shot
        self.ppm_scales = ppm_scales
        self.mpnn = MPNN2(256, no_pass, device_number)
        models.BatchNorm = BatchNorm

        self.device_no = device_number
        resnet = models.resnet50(pretrained=pretrained)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2,
                                    resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        reduce_dim = 256

        fea_dim = 1024 + 512

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim * 5, reduce_dim, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        # self.down_supp = nn.Sequential(
        #     nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(p=0.5)
        # )

        self.pyramid_bins = ppm_scales
        self.avgpool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(
                    nn.AdaptiveAvgPool2d(bin)
                )

        factor = 1
        mask_add_num = 1
        self.init_merge = []
        self.beta_conv = []
        self.inner_cls = []
        for bin in self.pyramid_bins:
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(reduce_dim * 2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
            ))
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True)
            ))
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(reduce_dim, classes, kernel_size=1)
            ))
        self.init_merge = nn.ModuleList(self.init_merge)
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.inner_cls = nn.ModuleList(self.inner_cls)

        self.compressor = nn.Sequential(
            nn.Conv2d(reduce_dim * len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True), nn.Dropout2d(p=0.5),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=(1, 1), padding=0, bias=True),
            nn.ReLU(inplace=True), nn.Dropout2d(p=0.5))

        self.res1 = nn.Sequential(nn.Conv2d(reduce_dim, reduce_dim, kernel_size=(3, 3), padding=1, bias=True),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(reduce_dim, reduce_dim, kernel_size=(3, 3), padding=1, bias=True),
                                  nn.ReLU(inplace=True))

        self.res2 = nn.Sequential(nn.Conv2d(reduce_dim, reduce_dim, kernel_size=(3, 3), padding=1, bias=True),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(reduce_dim, reduce_dim, kernel_size=(3, 3), padding=1, bias=True),
                                  nn.ReLU(inplace=True))

        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.layer6_0 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )

        self.layer6_1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )

        self.layer6_2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=6, dilation=6, bias=True),
                                      nn.ReLU(),
                                      nn.Dropout2d(p=0.5)
                                      )

        self.layer6_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=12, dilation=12, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )

        self.layer6_4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=18, dilation=18, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )



    def forward(self, x, s_x=torch.FloatTensor(1, 1, 3, 473, 473).cuda(3), s_y=torch.FloatTensor(1, 1, 473, 473).cuda(3),
                y=None):
        x_size = x.size()
        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        #   Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_0 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_0)
            query_feat_3 = self.layer3(query_feat_2)
            query_feat_4 = self.layer4(query_feat_3)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)

        #   Support Feature
        supp_feat_list = []
        final_supp_list = []
        mask_list = []
        for i in range(self.shot):
            mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:, i, :, :, :])
                supp_feat_0 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_0)
                supp_feat_3 = self.layer3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear',
                                     align_corners=True)
                supp_feat_4 = self.layer4(supp_feat_3 * mask)
                final_supp_list.append(supp_feat_4)

            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat = self.down_query(supp_feat)
            supp_feat = Weighted_GAP(supp_feat, mask)
            supp_feat_list.append(supp_feat)

        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
            q = query_feat_4
            s = tmp_supp_feat_4
            bsize, ch_sz, sp_sz, _ = q.size()[:]

            tmp_query = q
            tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            tmp_supp = s
            tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)
            tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

            similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
            similarity = similarity.max(1)[0].view(bsize, sp_sz * sp_sz)
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                    similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=(query_feat_3.size()[2], query_feat_3.size()[3]),
                                       mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)
        corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1)
        corr_query_mask = F.interpolate(corr_query_mask, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear',
                                        align_corners=True)

        if self.shot > 1:
            supp_feat = supp_feat_list[0]
            for i in range(1, len(supp_feat_list)):
                supp_feat += supp_feat_list[i]
            supp_feat /= len(supp_feat_list)

        # Support feature list o anki indekse ait tüm batchlerdeki embeddingleri tutar.
        # B,C,1,1 - 0shot B,C,1,1 - 1shot ... tüm hepsinin ortalamasıda her bir episode için masked global average
        # pooled supportların ortalamasını veriyor. Sonuç olarak B,C,1,1 ortalama maskeleri tutuyor.
        # PRIOR MASKLERIN boyutunun da B,1,60,60 olduğunu düşünüyorum.
        ## SAGNN Buradan itibaren ekleyebilirsin.

        out_list = []
        pyramid_feat_list = []

        for idx, tmp_bin in enumerate(self.pyramid_bins):
            if tmp_bin <= 1.0:
                bin = int(query_feat.shape[2] * tmp_bin)
                query_feat_bin = nn.AdaptiveAvgPool2d(bin)(query_feat)
            else:
                bin = tmp_bin
                query_feat_bin = self.avgpool_list[idx](query_feat)
            supp_feat_bin = supp_feat.expand(-1, -1, bin, bin)
            corr_mask_bin = F.interpolate(corr_query_mask, size=(bin, bin), mode='bilinear', align_corners=True)
            merge_feat_bin = torch.cat([query_feat_bin, supp_feat_bin, corr_mask_bin], 1)  # !
            merge_feat_bin = self.init_merge[idx](merge_feat_bin)
            merge_feat_bin = self.beta_conv[idx](merge_feat_bin)
            inner_out_bin = self.inner_cls[idx](merge_feat_bin)
            merge_feat_bin = F.interpolate(merge_feat_bin, size=(query_feat.size(2), query_feat.size(3)),
                                           mode='bilinear', align_corners=True)
            pyramid_feat_list.append(merge_feat_bin)
            out_list.append(inner_out_bin)

        # H = torch.stack(pyramid_feat_list, 1)
        H = self.mpnn(pyramid_feat_list)
        H = self.compressor(torch.cat(H, 1))
        H = self.res1(H) + H
        H = self.res2(H) + H


        # ASPP ekle
        global_feature = F.avg_pool2d(H, kernel_size=H.shape[-2:])
        global_feature = self.layer6_0(global_feature)
        global_feature = global_feature.expand(-1, -1, H.shape[2], H.shape[3])
        out = torch.cat([global_feature, self.layer6_1(H), self.layer6_2(H),
                         self.layer6_3(H), self.layer6_4(H)], dim=1)

        out = self.cls(out)

        #   Output Part
        if self.zoom_factor != 1:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            main_loss = self.criterion(out, y.long())
            aux_loss = torch.zeros_like(main_loss)

            for idx_k in range(len(out_list)):
                inner_out = out_list[idx_k]
                inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)
                aux_loss = aux_loss + self.criterion(inner_out, y.long())
            aux_loss = aux_loss / len(out_list)
            return out.max(1)[1], main_loss, aux_loss
        else:
            return out


if __name__ == "__main__":
    net = SAGNN(shot=5, no_pass=1, device_number=3)
    net.to("cuda:3")
    query_set = torch.rand(size=(4, 3, 473, 473), device="cuda:3")
    query_mask = torch.randint(low=0, high=1, size=(4, 473, 473), dtype=torch.long, device="cuda:3")
    support_set = torch.rand(size=(4, 5, 3, 473, 473), device="cuda:3")
    support_masks = torch.randint(low=0, high=1, size=(4, 5, 473, 473), device="cuda:3")
    a = net(query_set, support_set, support_masks, query_mask)
