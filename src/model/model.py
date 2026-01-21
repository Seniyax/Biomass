import torch
import torch.nn as nn
import timm




class Regressor(nn.Module):
    def __init__(self,model_name,pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained,num_classes=0)
        self.n_features = self.backbone.num_features
        self.n_combined = self.n_features *2

        def make_head():
            return nn.Sequential(
                nn.Linear(self.n_combined,self.n_combined//2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.n_combined//2, 1)
            )
        
        self.head_total = make_head()
        self.head_gdm = make_head()
        self.head_green = make_head()

    def forward(self, img_left:torch.Tensor,img_right:torch.Tensor):
        feat_left = self.backbone(img_left)
        feat_right = self.backbone(img_right)
        
        combined_feat = torch.cat([feat_left, feat_right], dim=1)

        out_total = self.head_total(combined_feat)
        out_gdm = self.head_gdm(combined_feat)
        out_green = self.head_green(combined_feat)

        return out_total, out_gdm, out_green
