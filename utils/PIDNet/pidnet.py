# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
try:
    from .model_utils import BasicBlock, Bottleneck, segmenthead, DAPPM, PAPPM, PagFM, Bag, Light_Bag
except:
    from model_utils import BasicBlock, Bottleneck, segmenthead, DAPPM, PAPPM, PagFM, Bag, Light_Bag
import logging

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False

class PIDNet(nn.Module):

    def __init__(self, m=2, n=3, num_classes=19, planes=64, ppm_planes=96, head_planes=128, augment=True, p3=0.0, p4=0.0, p5=0.0):
        super(PIDNet, self).__init__()
        self.augment = augment
        
        # Dropout
        
        self.drop3_P = nn.Dropout(p=p3)
        self.drop3_I = nn.Dropout(p=p3)
        self.drop3_D = nn.Dropout(p=p3)
        
        self.drop4_P = nn.Dropout(p=p4)
        self.drop4_I = nn.Dropout(p=p4)
        self.drop4_D = nn.Dropout(p=p4)
        
        self.drop5_P = nn.Dropout(p=p5)
        self.drop5_I = nn.Dropout(p=p5)
        self.drop5_D = nn.Dropout(p=p5)
        
        # I Branch
        self.conv1 =  nn.Sequential(
                          nn.Conv2d(3,planes,kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(planes,planes,kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                      )

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 =  self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)
        
        # P Branch
        self.compression3 = nn.Sequential(
                                          nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
                                          BatchNorm2d(planes * 2, momentum=bn_mom),
                                          )

        self.compression4 = nn.Sequential(
                                          nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
                                          BatchNorm2d(planes * 2, momentum=bn_mom),
                                          )
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)

        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)
        
        # D Branch
        if m == 2:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
            self.diff3 = nn.Sequential(
                                        nn.Conv2d(planes * 4, planes, kernel_size=3, padding=1, bias=False),
                                        BatchNorm2d(planes, momentum=bn_mom),
                                        )
            self.diff4 = nn.Sequential(
                                     nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                                     BatchNorm2d(planes * 2, momentum=bn_mom),
                                     )
            self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Light_Bag(planes * 4, planes * 4)
        else:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.layer4_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.diff3 = nn.Sequential(
                                        nn.Conv2d(planes * 4, planes * 2, kernel_size=3, padding=1, bias=False),
                                        BatchNorm2d(planes * 2, momentum=bn_mom),
                                        )
            self.diff4 = nn.Sequential(
                                     nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                                     BatchNorm2d(planes * 2, momentum=bn_mom),
                                     )
            self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4, planes * 4)
            
        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)
        
        # Prediction Head
        if self.augment:
            self.seghead_p = segmenthead(planes * 2, head_planes, num_classes)
            self.seghead_d = segmenthead(planes * 2, planes, 1) # num_classes)           

        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)
    
    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layer = block(inplanes, planes, stride, downsample, no_relu=True)
        
        return layer

    def forward(self, x):

        init_size = x.shape[-2:]

        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x))) # stage 2
        x_ = self.drop3_P(self.layer3_(x)) # P, stage 2->3
        x_d = self.drop3_D(self.layer3_d(x)) # D, stage 2->3
        
        # stage 3
        x = self.drop3_I(self.relu(self.layer3(x))) # I
        x_ = self.pag3(x_, self.compression3(x))
        x_d = x_d + F.interpolate(
                        self.diff3(x),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=algc)
        if self.augment:
            temp_p = x_
        
        # stage 4
        x = self.drop4_I(self.relu(self.layer4(x)))
        x_ = self.drop4_P(self.layer4_(self.relu(x_)))
        x_d = self.drop4_D(self.layer4_d(self.relu(x_d)))
        
        x_ = self.pag4(x_, self.compression4(x))
        x_d = x_d + F.interpolate(
                        self.diff4(x),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=algc)
        if self.augment:
            temp_d = x_d

        # stage 5
        x_ = self.drop5_P(self.layer5_(self.relu(x_)))
        x_d = self.drop5_D(self.layer5_d(self.relu(x_d)))
        x = F.interpolate(
                        self.spp(self.drop5_I(self.layer5(x))),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=algc)

        x_ = self.final_layer(self.dfm(x_, x, x_d)) # self.act1(self.final_layer(self.dfm(x_, x, x_d)))

        if self.augment:
            x_extra_p = self.seghead_p(temp_p) # self.act2(self.seghead_p(temp_p))
            x_extra_d = self.seghead_d(temp_d) # self.act3(self.seghead_d(temp_d))
            x_extra_p = F.interpolate(
                x_extra_p, size=init_size,
                mode='bilinear', align_corners=algc
            )
            x_ = F.interpolate(
                x_, size=init_size,
                mode='bilinear', align_corners=algc
            )
            x_extra_d = F.interpolate(
                x_extra_d, size=init_size,
                mode='bilinear', align_corners=algc
            )

            return [x_extra_p, x_, x_extra_d]
        else:
            return x_

def get_seg_model(name, num_classes,  p3=0.0, p4=0.0, p5=0.0): # model_pretrained, imgnet_pretrained=False,
    
    if 'pidnet_s' == name:
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=32, ppm_planes=96, head_planes=128, augment=True,
                      p3=p3, p4=p4, p5=p5)
    elif 'pidnet_m' == name:
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=64, ppm_planes=96, head_planes=128, augment=True,
                      p3=p3, p4=p4, p5=p5)
    elif 'pidnet_l' == name:
        model = PIDNet(m=3, n=4, num_classes=num_classes, planes=64, ppm_planes=112, head_planes=256, augment=True,
                      p3=p3, p4=p4, p5=p5)
    return model
#-----------------------------------------[PID-net]-----------------------------------------

#-------------------------------------------------------------------------------------------
# 
#     if imgnet_pretrained:
#         pretrained_state = torch.load(model_pretrained, map_location='cpu')['state_dict'] 
#         model_dict = model.state_dict()
#         pretrained_state = {k: v for k, v in pretrained_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
#         model_dict.update(pretrained_state)
#         msg = 'Loaded {} parameters!'.format(len(pretrained_state))
#         logging.info('Attention!!!')
#         logging.info(msg)
#         logging.info('Over!!!')
#         model.load_state_dict(model_dict, strict = False)
#     else:
#         pretrained_dict = torch.load(model_pretrained, map_location='cpu')
#         if 'state_dict' in pretrained_dict:
#             pretrained_dict = pretrained_dict['state_dict']
#         model_dict = model.state_dict()
#         pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
#         msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
#         logging.info('Attention!!!')
#         logging.info(msg)
#         logging.info('Over!!!')
    #     model_dict.update(pretrained_dict)
    #     model.load_state_dict(model_dict, strict = False)
    
    # return model

def get_pred_model(name, num_classes):
    
    if 's' in name:
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=32, ppm_planes=96, head_planes=128, augment=False)
    elif 'm' in name:
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=64, ppm_planes=96, head_planes=128, augment=False)
    else:
        model = PIDNet(m=3, n=4, num_classes=num_classes, planes=64, ppm_planes=112, head_planes=256, augment=False)
    
    return model

if __name__ == '__main__1':
    
    # Comment batchnorms here and in model_utils before testing speed since the batchnorm could be integrated into conv operation
    # (do not comment all, just the batchnorm following its corresponding conv layer)
    device = torch.device('cuda')
    model = get_pred_model(name='pidnet_s', num_classes=19)
    model.eval()
    model.to(device)
    iterations = None
    
    input = torch.randn(1, 3, 1024, 2048).cuda()
    with torch.no_grad():
        for _ in range(10):
            model(input)
    
        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)
    
        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)
    
    
    



def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    return (total, trainable)

if __name__ == "__main__":
    import torch.nn.functional as F
    param_size = dict()
    for vcf_mode in range(1, 4):
        print(f"VCF MODE: {vcf_mode}")
        for pidnet_mode in ['s', 'm', 'l']:
            print(f"PIDNET MODE: {pidnet_mode}")
            model = get_seg_model_vcf(f'pidnet_{pidnet_mode}', 17, 4, vcf_mode, p3=0.0, p4=0.0, p5=0.0)
            batch_num = 8
            channel_size = 3
            img_size = 1024
            x = torch.randn(batch_num, channel_size, img_size, img_size)
            pred = model(x)
            param_size[f"{pidnet_mode}-{vcf_mode}"] = count_parameters(model)
            if isinstance(pred, list):
                for i in pred:
                    print(f"Output: {i.size()}")
            else:
                print(f"Output: {pred.size()}")
    for k, v in param_size.items():
        print(k, ' : ', v)