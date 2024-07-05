import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn 
import numpy as np

import vision_transformer as vits
from torch.utils.data import Dataset
import json
from explanation_generator import Baselines
import random

device = torch.cuda.set_device(0) # set device
print('GPU:', torch.cuda.get_device_name(device))

# hyperparemeter setting
n_last_blocks = 4
pretrained_weights = "../SNNA/ckp/backbone_200.pth" # pretrained weights for backbone
checkpoint_key = "teacher"
arch = "vit_small"
patch_size = 8
num_labels = 4
classifier_weights_dir = "../SNNA/ckp" # pretrained weights for linear classifier
image_size = 360 # The image short side is resized to 360

#************************************************************#
# construct backbone model
backbone = vits.__dict__["vit_small"](patch_size=patch_size, num_classes=0)
embed_dim = backbone.embed_dim * n_last_blocks
# load backbone weights to evaluate
state_dict = torch.load(pretrained_weights, map_location="cpu")
if checkpoint_key is not None and checkpoint_key in state_dict:
    print(f"Take key {checkpoint_key} in provided checkpoint dict")
    state_dict = state_dict[checkpoint_key]
# remove `module.` prefix
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
# remove `backbone.` prefix induced by multicrop wrapper
state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
# load state dict for backbone
msg = backbone.load_state_dict(state_dict, strict=False)
print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
print(f"Model {arch} built.")

class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)
        # linear layer
        return self.linear(x)

# construct a linear classifier head
linear_classifier = LinearClassifier(embed_dim, num_labels)
# load pretrained weights for linear classifier
state_dict = torch.load(os.path.join(classifier_weights_dir, "classifier.pth.tar"), map_location="cpu")["state_dict"]
# remove `module.` prefix
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
# load state dict for linear classifier
linear_classifier.load_state_dict(state_dict, strict=True)

class Model(nn.Module):
    def __init__(self, backbone, head):
        super(Model, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, register_hook=None):
        x = x.unsqueeze(0).cuda() # (1, 3, w, h) add a batch dimension and send to GPU
        intermediate_output = self.backbone.get_intermediate_layers(x, n_last_blocks, register_hook=register_hook)
        output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        logits = self.head(output)
        return logits
    
# construct the model    
model = Model(backbone, linear_classifier)
model.cuda()
model.eval()

# load test dataset
dataset_path = '../SNNA/BDD-OIA/'

class myDataset(Dataset):
    def __init__(self, data_dir, label_dir, transform=None):
        self.label_dir = label_dir
        self.transform = transform
        
        self.images = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]
        self.labels = json.load(open(label_dir))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image = self.transform(image)

        image_name = self.images[idx].split('/')[-1]
        return image, torch.tensor(self.labels[image_name])
    
test_transform = transforms.Compose([
    transforms.Resize(image_size, interpolation=3), # 3 is bicubic interpolation
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
    
dataset_test = myDataset(os.path.join(dataset_path , "test"), os.path.join(dataset_path , "test.json"), transform=test_transform)


def multilabel_accuracy(output, target, threshold=0.5):
    """
    Computes the multilabel accuracy.
    """
    pred = output > threshold
    correct = pred.eq(target)
    return correct.float().mean()*100.

def reshape_attr(attribution, w_featmap, h_featmap, patch_size, w, h): 
    attribution = attribution.reshape(1, 1, w_featmap, h_featmap)
    attribution = torch.nn.functional.interpolate(attribution, scale_factor=patch_size, mode='bilinear')
    attribution = attribution.reshape(w, h).data.cpu().numpy()
    return attribution

def channel_wise_mask(mask,image):
    mask[0,:,:] = mask[0,:,:]*image[0,:,:].mean()
    mask[1,:,:] = mask[1,:,:]*image[1,:,:].mean()
    mask[2,:,:] = mask[2,:,:]*image[2,:,:].mean()
    return mask

def pixel_mask(attribution, img, mask_num, positive=True):
    flatten_attr = attribution.flatten() # return an array
    if positive:
        threshold = sorted(flatten_attr, reverse=True)[mask_num] # positive perturbation
        mask1 = (attribution <= threshold).astype(int) # mask the value less than threshold to 1
        mask2 = (attribution > threshold).astype(int) # mask the value greater than threshold to 1
    else:
        threshold = sorted(flatten_attr, reverse=False)[mask_num] # negative perturbation
        mask1 = (attribution >= threshold).astype(int) # mask the value greater than threshold to 1
        mask2 = (attribution < threshold).astype(int) # mask the value less than threshold to 1
    # masked_img = img * torch.tensor(mask1) + torch.tensor(mask2)*img.mean() # image pixel wize mean mask
    masked_img = img * torch.tensor(mask1) + channel_wise_mask(torch.tensor(mask2).expand(3,-1,-1), img) # channel wize mean mask
    return masked_img

def generate_explanation(attribution_generator, image, class_index= None):

    rawAttn = attribution_generator.rawAttn(image).detach()

    # rollout = attribution_generator.rollout(image, start_layer=0).detach()

    # gradient = attribution_generator.gradient(image, index=class_index).detach()

    att_gradient= attribution_generator.att_gradient(image, index=class_index).detach()

    generic_att = attribution_generator.generic_att(image, index=class_index).detach()

    norm_att = attribution_generator.norm_att(image, index=class_index).detach()

    IGradient = attribution_generator.IGradient(image, index=class_index, steps=20).detach()

    SGradient = attribution_generator.SGradient(image, index=class_index).detach()

    SGradient_normAtt = attribution_generator.SGradient_normAtt(image, index=class_index, magnitude=False).detach()

    return rawAttn, att_gradient, generic_att, norm_att, IGradient, SGradient, SGradient_normAtt
    
test_num = 3
random_idx = random.sample(range(len(dataset_test)), test_num)
print(random_idx)

# pixel_mask
rawAttn_degradation_accs = []
att_gradient_degradation_accs = []
generic_att_degradation_accs = []
norm_att_degradation_accs = []
IGradient_degradation_accs = []
SGradient_degradation_acc = []
SGradient_normAtt_degradation_accs = []

granularity = np.linspace(0, 0.2, 10)
attribution_generator = Baselines(model) 

for idx in random_idx:
    image = dataset_test[idx][0]
    label = dataset_test[idx][1]

    # make the image divisible by the patch size
    w, h = image.shape[1] - image.shape[1] % patch_size, image.shape[2] - image.shape[2] % patch_size
    img = image[:, :w, :h] 
    print(f"Image shape: {img.shape}")
    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size
    num_patches = w_featmap * h_featmap
    print(f"Feature map size: {w_featmap} x {h_featmap} = {num_patches} patches")

    output = model(img)
    print(f'output logits: {output}')
    # print(f'output probability: {torch.nn.functional.softmax(output, dim=-1)}')
    top_pred_label = np.argmax(output.cpu().data.numpy(), axis=-1)
    print(f"top_label: {top_pred_label}") # 0 for forward, 1 for slow down, 2 for left, 3 for right

    # get truc patch number
    mask_pixel_num = [int(g*w*h) for g in granularity]
    print(f"mask pixel number: {mask_pixel_num}")

    # generate explanation
    rawAttn, att_gradient, generic_att, norm_att, IGradient, SGradient, SGradient_normAtt = generate_explanation(attribution_generator, img, class_index=top_pred_label)

    # reshape Attribution to the same size as the input image
    rawAttn = reshape_attr(rawAttn, w_featmap, h_featmap, patch_size, w, h)
    att_gradient = reshape_attr(att_gradient, w_featmap, h_featmap, patch_size, w, h)
    generic_att = reshape_attr(generic_att, w_featmap, h_featmap, patch_size, w, h)
    norm_att = reshape_attr(norm_att, w_featmap, h_featmap, patch_size, w, h)
    IGradient = reshape_attr(IGradient, w_featmap, h_featmap, patch_size, w, h)
    SGradient = reshape_attr(SGradient, w_featmap, h_featmap, patch_size, w, h)
    SGradient_normAtt = reshape_attr(SGradient_normAtt, w_featmap, h_featmap, patch_size, w, h)


    rawAttn_pixel_degradation_accs = []
    att_gradient_pixel_degradation_accs = []
    generic_att_pixel_degradation_accs = []
    norm_att_pixel_degradation_accs = []
    IGradient_pixel_degradation_accs = []
    SGradient_pixel_degradation_accs = []
    SGradient_normAtt_pixel_degradation_accs = []

    for mask_num in mask_pixel_num[1:]: #exclude 0

        rawAttn_masked_img = pixel_mask(rawAttn, img, mask_num)
        rawAttn_masked_pred = model(rawAttn_masked_img)
        rawAttn_acc = multilabel_accuracy(rawAttn_masked_pred, label.cuda()).item()
        rawAttn_pixel_degradation_accs.append(rawAttn_acc)

        att_gradient_masked_img = pixel_mask(att_gradient, img, mask_num)
        att_gradient_masked_pred = model(att_gradient_masked_img)
        att_gradient_acc = multilabel_accuracy(att_gradient_masked_pred, label.cuda()).item()
        att_gradient_pixel_degradation_accs.append(att_gradient_acc)

        generic_att_masked_img = pixel_mask(generic_att, img, mask_num)
        generic_att_masked_pred = model(generic_att_masked_img)
        generic_att_acc = multilabel_accuracy(generic_att_masked_pred, label.cuda()).item()
        generic_att_pixel_degradation_accs.append(generic_att_acc)

        norm_att_masked_img = pixel_mask(norm_att, img, mask_num)
        norm_att_masked_pred = model(norm_att_masked_img)
        norm_att_acc = multilabel_accuracy(norm_att_masked_pred, label.cuda()).item()
        norm_att_pixel_degradation_accs.append(norm_att_acc)

        IGradient_masked_img = pixel_mask(IGradient, img, mask_num)
        IGradient_masked_pred = model(IGradient_masked_img)
        IGradient_acc = multilabel_accuracy(IGradient_masked_pred, label.cuda()).item()
        IGradient_pixel_degradation_accs.append(IGradient_acc)

        SGradient_masked_img = pixel_mask(SGradient, img, mask_num)
        SGradient_masked_pred = model(SGradient_masked_img)
        SGradient_acc = multilabel_accuracy(SGradient_masked_pred, label.cuda()).item()
        SGradient_pixel_degradation_accs.append(SGradient_acc)

        SGradient_normAtt_masked_img = pixel_mask(SGradient_normAtt, img, mask_num)
        SGradient_normAtt_masked_pred = model(SGradient_normAtt_masked_img)
        SGradient_normAtt_acc = multilabel_accuracy(SGradient_normAtt_masked_pred, label.cuda()).item()
        SGradient_normAtt_pixel_degradation_accs.append(SGradient_normAtt_acc)

    rawAttn_degradation_accs.append(rawAttn_pixel_degradation_accs)
    att_gradient_degradation_accs.append(att_gradient_pixel_degradation_accs)
    generic_att_degradation_accs.append(generic_att_pixel_degradation_accs)
    norm_att_degradation_accs.append(norm_att_pixel_degradation_accs)
    IGradient_degradation_accs.append(IGradient_pixel_degradation_accs)
    SGradient_degradation_acc.append(SGradient_pixel_degradation_accs)
    SGradient_normAtt_degradation_accs.append(SGradient_normAtt_pixel_degradation_accs)

rawAttn_avg_degradation_accs = np.round(np.mean(rawAttn_degradation_accs, axis=0),3)
print(f'rawAttn_avg_degradation_accs, {list(rawAttn_avg_degradation_accs)}')

att_gradient_avg_degradation_accs = np.round(np.mean(att_gradient_degradation_accs, axis=0),3)
print(f'att_gradient_avg_degradation_accs, {att_gradient_avg_degradation_accs}')

generic_att_avg_degradation_accs = np.round(np.mean(generic_att_degradation_accs, axis=0),3)
print(f'generic_att_avg_degradation_accs, {generic_att_avg_degradation_accs}')

norm_att_avg_degradation_accs = np.round(np.mean(norm_att_degradation_accs, axis=0),3)
print(f'norm_att_avg_degradation_accs, {norm_att_avg_degradation_accs}')

IGradient_avg_degradation_accs = np.round(np.mean(IGradient_degradation_accs, axis=0),3)
print(f'IGradient_avg_degradation_accs, {IGradient_avg_degradation_accs}')

SGradient_avg_degradation_accs = np.round(np.mean(SGradient_degradation_acc, axis=0),3)
print(f'SGradient_avg_degradation_accs, {SGradient_avg_degradation_accs}')

SGradient_normAtt_avg_degradation_accs = np.round(np.mean(SGradient_normAtt_degradation_accs, axis=0),3)
print(f'SGradient_normAtt_avg_degradation_accs, {SGradient_normAtt_avg_degradation_accs}')

