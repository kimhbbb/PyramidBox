import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGG_Base_Conv_layers(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.load_pretrained_layers()

    def forward(self, image):
        out = F.relu(self.conv1_1(image)) 
        out = F.relu(self.conv1_2(out))
        out = self.pool1(out)

        out = F.relu(self.conv2_1(out))  
        out = F.relu(self.conv2_2(out))  
        out = self.pool2(out)  

        out = F.relu(self.conv3_1(out))  
        out = F.relu(self.conv3_2(out))  
        out = F.relu(self.conv3_3(out)) 
        conv3_3_feats = out # (N, 256, 160, 160)
        out = self.pool3(out) 

        out = F.relu(self.conv4_1(out))  
        out = F.relu(self.conv4_2(out))  
        out = F.relu(self.conv4_3(out))  
        conv4_3_feats = out # (N, 512, 80, 80)
        out = self.pool4(out)  

        out = F.relu(self.conv5_1(out))  
        out = F.relu(self.conv5_2(out))  
        out = F.relu(self.conv5_3(out))
        conv5_3_feats = out # (N, 512, 40, 40)
        out = self.pool5(out) 

        out = F.relu(self.conv6(out))  

        conv_fc7_feats = F.relu(self.conv7(out)) # (N, 1024, 20, 20)

        return conv3_3_feats, conv4_3_feats, conv5_3_feats, conv_fc7_feats

    def load_pretrained_layers(self):
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        for i, param in enumerate(param_names[:-4]): 
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].reshape(4096, 512, 7, 7)  
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias'] 
        state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3]) 
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])  

        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].reshape(4096, 4096, 1, 1)  
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  
        state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None]) 
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4]) 

        self.load_state_dict(state_dict)

class Extra_Conv_layers(nn.Module): 
    def __init__(self, in_channels):
        super().__init__()
        self.conv6_1 = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.conv6_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv7_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv7_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

    def forward(self, conv_fc7_feats):
        out = F.relu(self.conv6_1(conv_fc7_feats)) 
        out = F.relu(self.conv6_2(out)) 
        conv6_2_feats = out # (N, 512, 10, 10)

        out = F.relu(self.conv7_1(out)) 
        out = F.relu(self.conv7_2(out)) 
        conv7_2_feats = out # (N, 256, 5, 5)

        return conv6_2_feats, conv7_2_feats
    
class LFPN_CPM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_lfpn_0 = nn.Conv2d(1024, 512, kernel_size=1)
        self.conv_lfpn_0_1 = nn.Conv2d(512, 512, 1, 1, 0)
        self.conv_lfpn_1 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv_lfpn_1_1 = nn.Conv2d(512, 512, 1, 1, 0)
        self.conv_lfpn_2 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv_lfpn_2_1 = nn.Conv2d(256, 256, 1, 1, 0)

        self.cpm_0 = self.cpm_block(256) 
        self.cpm_1 = self.cpm_block(512) 
        self.cpm_2 = self.cpm_block(512) 
        self.cpm_3 = self.cpm_block(1024) 
        self.cpm_4 = self.cpm_block(512) 
        self.cpm_5 = self.cpm_block(256) 

    def forward(self, conv3_3_feats, conv4_3_feats, conv5_3_feats, conv_fc7_feats, conv6_2_feats, conv7_2_feats):
        # Low-level Feature Pyramid Layars
        lfpn_0 = self.conv_lfpn_0_1(conv5_3_feats) * F.interpolate(self.conv_lfpn_0(conv_fc7_feats), # (N, 512, 40, 40)
                                                                   scale_factor=2, mode='bilinear', align_corners=False)
        lfpn_1 = self.conv_lfpn_1_1(conv4_3_feats) * F.interpolate(self.conv_lfpn_1(lfpn_0), # (N, 512, 80, 80)
                                                                   scale_factor=2, mode='bilinear', align_corners=False)
        lfpn_2 = self.conv_lfpn_2_1(conv3_3_feats) * F.interpolate(self.conv_lfpn_2(lfpn_1), # (N, 256, 160, 160)
                                                                   scale_factor=2, mode='bilinear', align_corners=False)

        # Context-sensitive Predict Layers
        # CPM에서 사용할 feature 추출.
        cpm_0 = self.cpm_0(lfpn_2)
        cpm_1 = self.cpm_1(lfpn_1)
        cpm_2 = self.cpm_2(lfpn_0)
        cpm_3 = self.cpm_3(conv_fc7_feats)
        cpm_4 = self.cpm_4(conv6_2_feats)
        cpm_5 = self.cpm_5(conv7_2_feats)

        return cpm_0, cpm_1, cpm_2, cpm_3, cpm_4, cpm_5 
    
    def cpm_block(self, in_channels):
        return CPMBlock(in_channels)

class CPMBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.residual1_1 = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.conv1_1_1 = nn.Conv2d(in_channels, 1024, kernel_size=1)
        self.conv1_1_2 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv1_1_3 = nn.Conv2d(256, 256, kernel_size=1)

        self.residual1_2 = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.conv1_2_1 = nn.Conv2d(in_channels, 1024, kernel_size=1)
        self.conv1_2_2 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv1_2_3 = nn.Conv2d(256, 256, kernel_size=1)
        
        self.residual2_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv2_1_1 = nn.Conv2d(256, 1024, kernel_size=1)
        self.conv2_1_2 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv2_1_3 = nn.Conv2d(256, 128, kernel_size=1)

        self.residual2_2 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv2_2_1 = nn.Conv2d(256, 1024, kernel_size=1)
        self.conv2_2_2 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv2_2_3 = nn.Conv2d(256, 128, kernel_size=1)

        self.residual3 = nn.Conv2d(128, 128, kernel_size=1)
        self.conv3_1 = nn.Conv2d(128, 1024, kernel_size=1)
        self.conv3_2 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv3_3 = nn.Conv2d(256, 128, kernel_size=1)
    
    def forward(self, feats):
        residual1_1 = self.residual1_1(feats)
        cpm1_1 = self.conv1_1_1(feats)
        cpm1_1 = self.conv1_1_2(cpm1_1)
        cpm1_1 = self.conv1_1_3(cpm1_1)
        cpm1_1 = torch.add(residual1_1, cpm1_1) 
        cpm1_1 = F.relu(cpm1_1)
        # print("cpm1_1: {}".format(cpm1_1.shape)) # (N, 256, 160, 160) / (80, 80) / (40, 40) / (20, 20) / (10, 10) / (5, 5)

        residual1_2 = self.residual1_2(feats) 
        cpm1_2 = self.conv1_2_1(feats)
        cpm1_2 = self.conv1_2_2(cpm1_2)
        cpm1_2 = self.conv1_2_3(cpm1_2)
        cpm1_2 = torch.add(residual1_2, cpm1_2) 
        cpm1_2 = F.relu(cpm1_2)
        # print("cpm1_2: {}".format(cpm1_2.shape)) # (N, 256, 160, 160) / (80, 80) / (40, 40) / (20, 20) / (10, 10) / (5, 5)

        residual2_1 = self.residual2_1(cpm1_2)
        cpm2_1 = self.conv2_1_1(cpm1_2)
        cpm2_1 = self.conv2_1_2(cpm2_1)
        cpm2_1 = self.conv2_1_3(cpm2_1)
        cpm2_1 = torch.add(residual2_1, cpm2_1) 
        cpm2_1 = F.relu(cpm2_1)
        # print("cpm2_1: {}".format(cpm2_1.shape)) # (N, 128, 160, 160) / (80, 80) / (40, 40) / (20, 20) / (10, 10) / (5, 5)

        residual2_2 = self.residual2_2(cpm1_2)
        cpm2_2 = self.conv2_2_1(cpm1_2)
        cpm2_2 = self.conv2_2_2(cpm2_2)
        cpm2_2 = self.conv2_2_3(cpm2_2)
        cpm2_2 = torch.add(residual2_2, cpm2_2) 
        cpm2_2 = F.relu(cpm2_2)
        # print("cpm2_2: {}".format(cpm2_2.shape)) # (N, 128, 160, 160) / (80, 80) / (40, 40) / (20, 20) / (10, 10) / (5, 5)

        residual3 = self.residual3(cpm2_2)
        cpm3_1 = self.conv3_1(cpm2_2)
        cpm3_1 = self.conv3_2(cpm3_1)
        cpm3_1 = self.conv3_3(cpm3_1)
        cpm3_1 = torch.add(residual3, cpm3_1) 
        cpm3_1 = F.relu(cpm3_1)
        # print("cpm3_1: {}".format(cpm3_1.shape)) # (N, 128, 160, 160) / (80, 80) / (40, 40) / (20, 20) / (10, 10) / (5, 5)

        cpm_out = torch.cat([cpm1_1, cpm2_1, cpm3_1], dim=1)
        cpm_out = F.relu(cpm_out)
        # print("cpm_out: {}".format(cpm_out.shape)) # (N, 512, 160, 160) / (80, 80) / (40, 40) / (20, 20) / (10, 10) / (5, 5)

        return cpm_out

class CPM_Predict(nn.Module):
    def __init__(self):
        super().__init__()
        # body 제외.
        self.loc_conv0 = nn.Conv2d(512, 8, kernel_size=3, padding=1)
        self.loc_conv1 = nn.Conv2d(512, 8, kernel_size=3, padding=1)
        self.loc_conv2 = nn.Conv2d(512, 8, kernel_size=3, padding=1)
        self.loc_conv3 = nn.Conv2d(512, 8, kernel_size=3, padding=1)
        self.loc_conv4 = nn.Conv2d(512, 8, kernel_size=3, padding=1)
        self.loc_conv5 = nn.Conv2d(512, 8, kernel_size=3, padding=1)

        self.conf_conv0 = nn.Conv2d(512, 6, kernel_size=3, padding=1)
        self.conf_conv1 = nn.Conv2d(512, 6, kernel_size=3, padding=1)
        self.conf_conv2 = nn.Conv2d(512, 6, kernel_size=3, padding=1)
        self.conf_conv3 = nn.Conv2d(512, 6, kernel_size=3, padding=1)
        self.conf_conv4 = nn.Conv2d(512, 6, kernel_size=3, padding=1)
        self.conf_conv5 = nn.Conv2d(512, 6, kernel_size=3, padding=1)

        self.init_conv2d()

    def forward(self, cpm_0, cpm_1, cpm_2, cpm_3, cpm_4, cpm_5):
        batch_size = cpm_0.size(0)

        loc_conv0 = self.loc_conv0(cpm_0) # (N, 8, 160, 160)
        loc_conv0 = loc_conv0.permute(0, 2, 3, 1).contiguous() # (N, 160, 160, 8)
        loc_conv0 = loc_conv0.view(batch_size, -1, 8) # (N, 25600, 8)
        face_loc0, head_loc0 = loc_conv0.split(4, dim=-1) # 각각 (N, 25600, 4)

        loc_conv1 = self.loc_conv1(cpm_1) # (N, 8, 80, 80)
        loc_conv1 = loc_conv1.permute(0, 2, 3, 1).contiguous() # (N, 80, 80, 8)
        loc_conv1 = loc_conv1.view(batch_size, -1, 8) # (N, 6400, 8)
        face_loc1, head_loc1 = loc_conv1.split(4, dim=-1) # 각각 (N, 6400, 4)

        loc_conv2 = self.loc_conv2(cpm_2) # (N, 8, 40, 40)
        loc_conv2 = loc_conv2.permute(0, 2, 3, 1).contiguous() # (N, 40, 40, 8)
        loc_conv2 = loc_conv2.view(batch_size, -1, 8) # (N, 1600, 8)
        face_loc2, head_loc2 = loc_conv2.split(4, dim=-1) # 각각 (N, 1600, 4)

        loc_conv3 = self.loc_conv3(cpm_3) # (N, 8, 20, 20)
        loc_conv3 = loc_conv3.permute(0, 2, 3, 1).contiguous() # (N, 20, 20, 8)
        loc_conv3 = loc_conv3.view(batch_size, -1, 8) # (N, 400, 8)
        face_loc3, head_loc3 = loc_conv3.split(4, dim=-1) # 각각 (N, 400, 4)

        loc_conv4 = self.loc_conv4(cpm_4) # (N, 8, 10, 10)
        loc_conv4 = loc_conv4.permute(0, 2, 3, 1).contiguous() # (N, 10, 10, 8)
        loc_conv4 = loc_conv4.view(batch_size, -1, 12) # (N, 100, 8)
        face_loc4, head_loc4 = loc_conv4.split(4, dim=-1) # 각각 (N, 100, 4)

        loc_conv5 = self.loc_conv5(cpm_5) # (N, 8, 5, 5)
        loc_conv5 = loc_conv5.permute(0, 2, 3, 1).contiguous() # (N, 5, 5, 8)
        loc_conv5 = loc_conv5.view(batch_size, -1, 8) # (N, 25, 8)
        face_loc5, head_loc5 = loc_conv5.split(4, dim=-1) # 각각 (N, 25, 4)


        conf_conv0 = self.conf_conv0(cpm_0) # (N, 6, 160, 160)
        conf_conv0 = conf_conv0.permute(0, 2, 3, 1).contiguous() # (N, 160, 160, 6)
        conf_conv0 = conf_conv0.view(batch_size, -1, 6) # (N, 25600, 6)
        face_conf0, head_conf0 = conf_conv0.split([4, 2], dim=-1) # (N, 25600, 4) / (N, 25600, 2)
        face_conf0_pos = face_conf0[:,:,:1]
        face_conf0_neg, _ = torch.max(face_conf0[:,:,1:], dim=-1, keepdim=True)
        face_conf0 = torch.cat([face_conf0_pos, face_conf0_neg], dim=-1) # (N, 25600, 2)

        conf_conv1 = self.conf_conv1(cpm_1) # (N, 6, 80, 80)
        conf_conv1 = conf_conv1.permute(0, 2, 3, 1).contiguous() # (N, 80, 80, 6)
        conf_conv1 = conf_conv1.view(batch_size, -1, 6) # (N, 6400, 6)
        face_conf1, head_conf1, body_conf1 = conf_conv1.split([4, 2], dim=-1) # (N, 6400, 4) / (N, 6400, 2)
        face_conf1_pos, _ = torch.max(face_conf1[:,:,:3], dim=-1, keepdim=True)
        face_conf1_neg = face_conf1[:,:,3:]
        face_conf1 = torch.cat([face_conf1_pos, face_conf1_neg], dim=-1) # (N, 6400, 2)
 
        conf_conv2 = self.conf_conv2(cpm_2) # (N, 6, 40, 40)
        conf_conv2 = conf_conv2.permute(0, 2, 3, 1).contiguous() # (N, 40, 40, 6)
        conf_conv2 = conf_conv2.view(batch_size, -1, 6) # (N, 1600, 6)
        face_conf2, head_conf2, body_conf2 = conf_conv2.split([4, 2], dim=-1) # (N, 1600, 4) / (N, 1600, 2)
        face_conf2_pos, _ = torch.max(face_conf2[:,:,:3], dim=-1, keepdim=True)
        face_conf2_neg = face_conf2[:,:,3:]
        face_conf2 = torch.cat([face_conf2_pos, face_conf2_neg], dim=-1) # (N, 1600, 2)

        conf_conv3 = self.conf_conv3(cpm_3) # (N, 6, 20, 20)
        conf_conv3 = conf_conv3.permute(0, 2, 3, 1).contiguous() # (N, 20, 20, 6)
        conf_conv3 = conf_conv3.view(batch_size, -1, 6) # (N, 400, 6)
        face_conf3, head_conf3, body_conf3 = conf_conv3.split([4, 2], dim=-1) # (N, 400, 4) / (N, 400, 2)
        face_conf3_pos, _ = torch.max(face_conf3[:,:,:3], dim=-1, keepdim=True)
        face_conf3_neg = face_conf3[:,:,3:]
        face_conf3 = torch.cat([face_conf3_pos, face_conf3_neg], dim=-1) # (N, 400, 2)

        conf_conv4 = self.conf_conv4(cpm_4) # (N, 6, 10, 10)
        conf_conv4 = conf_conv4.permute(0, 2, 3, 1).contiguous() # (N, 10, 10, 6)
        conf_conv4 = conf_conv4.view(batch_size, -1, 6) # (N, 100, 6)
        face_conf4, head_conf4, body_conf4 = conf_conv4.split([4, 2], dim=-1) # (N, 100, 4) / (N, 100, 2)
        face_conf4_pos, _ = torch.max(face_conf4[:,:,:3], dim=-1, keepdim=True)
        face_conf4_neg = face_conf4[:,:,3:]
        face_conf4 = torch.cat([face_conf4_pos, face_conf4_neg], dim=-1) # (N, 100, 2)
        
        conf_conv5 = self.conf_conv5(cpm_5) # (N, 6, 5, 5)
        conf_conv5 = conf_conv5.permute(0, 2, 3, 1).contiguous() # (N, 5, 5, 6)
        conf_conv5 = conf_conv5.view(batch_size, -1, 6) # (N, 25, 6)
        face_conf5, head_conf5 = conf_conv5.split([4, 2], dim=-1) # (N, 25, 4) / (N, 25, 2)
        face_conf5_pos, _ = torch.max(face_conf5[:,:,:3], dim=-1, keepdim=True)
        face_conf5_neg = face_conf5[:,:,3:]
        face_conf5 = torch.cat([face_conf5_pos, face_conf5_neg], dim=-1) # (N, 25, 2)

        face_locs = torch.cat([face_loc0, face_loc1, face_loc2, face_loc3, face_loc4, face_loc5], dim=1) # (N, 34125, 4)
        head_locs = torch.cat([head_loc0, head_loc1, head_loc2, head_loc3, head_loc4, head_loc5], dim=1) # (N, 34125, 4)

        face_scores = torch.cat([face_conf0, face_conf1, face_conf2, face_conf3, face_conf4, face_conf5], dim=1) # (N, 34125, 2)
        head_scores = torch.cat([head_conf0, head_conf1, head_conf2, head_conf3, head_conf4, head_conf5], dim=1) # (N, 34125, 2)

        return face_locs, head_locs, face_scores, head_scores

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

class PyramidBox(nn.Module):
    def __init__(self):
        super().__init__()
        # architecture
        self.vgg = VGG_Base_Conv_layers() 
        self.extra = Extra_Conv_layers(1024)
        # prediciton
        self.lfpn_cpm = LFPN_CPM()
        self.cpm_predict = CPM_Predict()

        self.face_prior_cxcy, self.head_prior_cxcy = self.create_prior_boxes()

    def forward(self, image):
        # (N, 256, 160, 160) / (N, 512, 80, 80) / (N, 512, 40, 40) / (N, 1024, 20, 20) / (N, 512, 10, 10) / (N, 256, 5, 5)
        conv3_3_feats, conv4_3_feats, conv5_3_feats, conv_fc7_feats = self.vgg(image)
        conv6_2_feats, conv7_2_feats = self.extra(conv_fc7_feats)

        # (N, 512, 160, 160) / (80, 80) / (40, 40) / (20, 20) / (10, 10) / (5, 5)
        cpm_0, cpm_1, cpm_2, cpm_3, cpm_4, cpm_5 = self.lfpn_cpm(conv3_3_feats, conv4_3_feats, conv5_3_feats, 
                                                                 conv_fc7_feats, conv6_2_feats, conv7_2_feats)
        face_locs, head_locs, face_scores, head_scores = self.cpm_predict(cpm_0, cpm_1, cpm_2, cpm_3, cpm_4, cpm_5)

        return face_locs, head_locs, face_scores, head_scores

    def create_prior_boxes(self):
        # aspect ratio = 1
        # face, head prior boxes 분리. scale 적용하자..(x2)
        
        image_size = 640
        feature_size = [160, 80, 40, 20, 10, 5]
        anchor_size = [16, 32, 64, 128, 256, 512]
        head_scale = 2

        face_prior_boxes = []
        head_prior_boxes = []

        for k in range(6):
            for i in range(feature_size[k]):
                for j in range(feature_size[k]):
                    cx = (j + 0.5) / feature_size[k]
                    cy = (i + 0.5) / feature_size[k]
                    f_w = anchor_size[k] / image_size
                    f_h = anchor_size[k] / image_size

                    h_w = (anchor_size[k] * head_scale / image_size) 
                    h_h = (anchor_size[k] * head_scale / image_size) 
                    
                    face_prior_boxes.append([cx, cy, f_w, f_h])
                    head_prior_boxes.append([cx, cy, h_w, h_h])

        face_prior_boxes = torch.FloatTensor(face_prior_boxes)
        face_prior_boxes.clamp_(0, 1)

        head_prior_boxes = torch.FloatTensor(head_prior_boxes)
        head_prior_boxes.clamp_(0, 1)

        return face_prior_boxes, head_prior_boxes

class PyramidBoxLoss(nn.Module):
    def __init__(self, priors_cxcy, threshold = 0.5, supervised = True):
        super().__init__()
        self.priors_cxcy = priors_cxcy
        self.prior_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.supervised = supervised # face anchor인지 아닌지 판단.

        self.smooth_l1 = nn.L1Loss()
        self.cross_entorpy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        batch_size = predicted_locs.size(0)
        
        true_locs = torch.zeros([batch_size, 1, 4], dtype=torch.float).to(device)
        true_classes = torch.zeros([batch_size, 1], dtype=float.long).to(device)

        for i in range(batch_size):
            n_faces = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i], self.priors_cxcy)


            





            pass

        

        return loc_loss, conf_loss

if __name__ == '__main__':
    net = PyramidBox()
    image = torch.randn(1, 3, 640, 640)
    net(image)