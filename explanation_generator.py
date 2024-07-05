import torch
import numpy as np
from numpy import *

# compute rollout between attention layers
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention

def avg_heads(att, grad):
    att = att.reshape(-1, att.shape[-2], att.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * att # * for element-wise multiplicationm , @ for matrix multiplication
    cam = cam.clamp(min=0).mean(dim=0) # clamp the negative value to 0, then take the mean
    return cam

def attn_weighted_norm(attn, v, vproj, dim, num_heads):
    with torch.no_grad():
        # value_layer is converted to (batch, seq_length, num_heads, 1, head_size)
        # print(f'v: {v.size()}')
        value_layer = v.permute(0, 2, 1, 3).contiguous()
        value_shape = value_layer.size()
        value_layer = value_layer.view(value_shape[:-1] + (1, value_shape[-1],))
        
        # proj_weight is converted to (num_heads, head_size, all_head_size)
        proj_weight = vproj.weight
        proj_weight = proj_weight.view(dim, num_heads, dim // num_heads)
        proj_weight = proj_weight.permute(1, 2, 0).contiguous()
        
        # Make transformed vectors f(x) from Value vectors (value_layer) and weight matrix (dense).
        transformed_layer = value_layer.matmul(proj_weight)
        transformed_shape = transformed_layer.size() #(batch, seq_length, num_heads, 1, all_head_size)
        transformed_layer = transformed_layer.view(transformed_shape[:-2] + (transformed_shape[-1],))
        transformed_layer = transformed_layer.permute(0, 2, 1, 3).contiguous()

        transformed_norm = torch.linalg.norm(transformed_layer, dim=-1) #(batch, num_heads, seq_length)

        # print('Memory usage: {} (GB)'.format(torch.cuda.memory_allocated() / 1e9))

        # Make weighted vectors α||f(x)|| from transformed vectors (transformed_layer) and attention weights (attention_probs).
        weighted_norm = torch.einsum('bhns,bhn->bhns', attn, transformed_norm) #(batch, num_heads, seq_length, seq_length, all_head_size)
        return weighted_norm
    
# def all_norm(attn, v, vproj, dim, num_heads):
#     with torch.no_grad():
#         # value_layer is converted to (batch, seq_length, num_heads, 1, head_size)
#         # print(f'v: {v.size()}')
#         value_layer = v.permute(0, 2, 1, 3).contiguous()
#         value_shape = value_layer.size()
#         value_layer = value_layer.view(value_shape[:-1] + (1, value_shape[-1],))
        
#         # proj_weight is converted to (num_heads, head_size, all_head_size)
#         proj_weight = vproj.weight
#         proj_weight = proj_weight.view(dim, num_heads, dim // num_heads)
#         proj_weight = proj_weight.permute(1, 2, 0).contiguous()
        
#         # Make transformed vectors f(x) from Value vectors (value_layer) and weight matrix (dense).
#         transformed_layer = value_layer.matmul(proj_weight)
#         transformed_shape = transformed_layer.size() #(batch, seq_length, num_heads, 1, all_head_size)
#         transformed_layer = transformed_layer.view(transformed_shape[:-2] + (transformed_shape[-1],))
#         transformed_layer = transformed_layer.permute(0, 2, 1, 3).contiguous()

#         # the below code of einsum will out o fmemory, as this is a huge tensor (CUDA out of memory. Tried to allocate 111.30 GiB )
#         weighted_norm = torch.einsum('bhns,bhnd->bhnsd', attn, transformed_layer) #(batch, num_heads, seq_length, seq_length, all_head_size) 

#         weighted_norm = torch.linalg.norm(weighted_norm, dim=-1) #(batch, num_heads, seq_length, seq_length)
#         # print(f'weighted_norm: {weighted_norm.size()}')

#         return weighted_norm


class Baselines:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.dim = model.backbone.embed_dim
        self.num_heads = model.backbone.num_heads

    def rawAttn(self, input):
        self.model(input)
        blocks = self.model.backbone.blocks
        last_attn= blocks[-1].attn.get_attention_map()
        R = (last_attn.sum(dim=1) / last_attn.shape[1])
        print(f'rawAttn R: {R.size()}')
        return R[:, 0, 1:]
        
    def rollout(self, input, start_layer=0):
        self.model(input)
        blocks = self.model.backbone.blocks
        all_layer_attentions = []
        for blk in blocks:
            attn_heads = blk.attn.get_attention_map()
            avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
            all_layer_attentions.append(avg_heads)
        R = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
        print(f'rollout R: {R.size()}')
        return R[:,0, 1:]
    
    def gradient(self, input, index=None):
        output = self.model(input, register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"top_class: {index}") # 0 for forward, 1 for slow down, 2 for left, 3 for right

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        R = 0 
        for blk in self.model.backbone.blocks:
            grad = blk.attn.get_attn_gradients()
            R += grad.sum(dim=1) / grad.shape[1]
        print(f'gradient R: {R.size()}')
        return R[:,0, 1:]
    
    def att_gradient(self, input, index=None):
        output = self.model(input, register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"top_class: {index}") # 0 for forward, 1 for slow down, 2 for left, 3 for right

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        R = 0 
        for blk in self.model.backbone.blocks:
            grad = blk.attn.get_attn_gradients()
            att = blk.attn.get_attention_map()
            cam = avg_heads(att, grad)
            R += cam
        print(f'att_gradient R: {R.size()}')
        return R[0, 1:]
    
    def generic_att(self, input, index=None):
        output = self.model(input, register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"top_class: {index}") # 0 for forward, 1 for slow down, 2 for left, 3 for right

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        
        # num_tokens = w_featmap * h_featmap + 1 # +1 for the [CLS] token
        num_tokens = self.model.backbone.blocks[0].attn.get_attention_map().shape[-1]
        
        R = torch.eye(num_tokens, num_tokens).cuda() # initialize R^0 as the identity matrix
        for blk in self.model.backbone.blocks:
            grad = blk.attn.get_attn_gradients()
            att = blk.attn.get_attention_map()
            cam = avg_heads(att, grad)
            R += torch.matmul(cam, R)

        print(f'generic_att R: {R.size()}')
        return R[0, 1:]
    
    def norm_att(self, input, index=None):
        output = self.model(input, register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"top_class: {index}") # 0 for forward, 1 for slow down, 2 for left, 3 for right

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        # num_tokens = w_featmap * h_featmap + 1 # +1 for the [CLS] token
        num_tokens = self.model.backbone.blocks[0].attn.get_attention_map().shape[-1]
        
        R = I = torch.eye(num_tokens, num_tokens).cuda() # initialize R^0 as the identity matrix
        for blk in self.model.backbone.blocks:
            att = blk.attn.get_attention_map()
            grad = blk.attn.get_attn_gradients()

            # weighted_norm = attn_weighted_norm(att, blk.attn.v, blk.attn.proj, blk.attn.dim, blk.attn.num_heads)
            weighted_norm = attn_weighted_norm(att, blk.attn.v, blk.attn.proj, self.dim, self.num_heads)

            cam = avg_heads(weighted_norm, grad) + I
            R = torch.matmul(cam, R)
        print(f'norm_att R: {R.size()}')
        return R[0, 1:]
    
    def IGradient(self, input, index=None, steps = 20):
        output = self.model(input, register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"top_class: {index}") # 0 for forward, 1 for slow down, 2 for left, 3 for right

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        
        # num_tokens = w_featmap * h_featmap + 1 # +1 for the [CLS] token
        num_tokens = self.model.backbone.blocks[0].attn.get_attention_map().shape[-1]
        
        R = torch.eye(num_tokens, num_tokens).cuda() # initialize R^0 as the identity matrix
        for blk in self.model.backbone.blocks:
            grad = blk.attn.get_attn_gradients()
            att = blk.attn.get_attention_map()
            cam = avg_heads(att, grad)
            R += torch.matmul(cam, R)
        
        # integrated gradient     
        x = self.model.backbone.blocks[0].attn.get_input()
        B, N, C = x.shape # (1, 3601, 384)  
        total_gradients = torch.zeros(B, self.num_heads, num_tokens, num_tokens).cuda()
        for alpha in np.linspace(0, 1, steps):        
            # forward propagation
            data_scaled = input * alpha

            # backward propagation
            output = self.model(data_scaled, register_hook=True)
            one_hot = np.zeros((B, output.size()[-1]), dtype=np.float32)
            one_hot[np.arange(B), index] = 1
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * output)

            self.model.zero_grad()
            one_hot.backward(retain_graph=True)

            # cal grad
            gradients = self.model.backbone.blocks[-1].attn.get_attn_gradients()
            total_gradients += gradients        
       
        W_state = (total_gradients / steps).clamp(min=0).mean(1).reshape(B, num_tokens, num_tokens)
            
        R = W_state * R.unsqueeze(0)
        print(f'IGradient R: {R.size()}')
        return R[:, 0, 1:]

    def SGradient(self, input, index=None ,stdev_spread=.15, nsamples=25, magnitude=False):
        """Returns a mask that is smoothed with the SmoothGrad method.

        Args:
            magnitude: If true, computes the sum of squares of gradients instead of
                    just the sum. Defaults to true.
        """
        
        output = self.model(input, register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"top_class: {index}") # 0 for forward, 1 for slow down, 2 for left, 3 for right

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        
        # num_tokens = w_featmap * h_featmap + 1 # +1 for the [CLS] token
        num_tokens = self.model.backbone.blocks[0].attn.get_attention_map().shape[-1]
        
        R = torch.eye(num_tokens, num_tokens).cuda() # initialize R^0 as the identity matrix
        for blk in self.model.backbone.blocks:
            grad = blk.attn.get_attn_gradients()
            att = blk.attn.get_attention_map()
            cam = avg_heads(att, grad)
            R += torch.matmul(cam, R)
        
        # smooth gradient     
        x = self.model.backbone.blocks[0].attn.get_input()
        B = x.shape[0] 
        # c, w, h = input.shape

        stdev = stdev_spread * (torch.max(input) - torch.min(input))
        total_gradients = torch.zeros((B, self.num_heads, num_tokens, num_tokens), dtype=torch.float32).cuda()

        for _ in range(nsamples):
            noise = torch.normal(0, stdev, input.shape) # create ramdom gaussian noise with mean 0 and std stdev
            x_plus_noise = input + noise

            # backward propagation
            output = self.model(x_plus_noise.cuda(), register_hook=True)
            one_hot = np.zeros((B, output.size()[-1]), dtype=np.float32)
            one_hot[np.arange(B), index] = 1
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * output)

            self.model.zero_grad()
            one_hot.backward(retain_graph=True)

            # cal grad
            gradients = self.model.backbone.blocks[-1].attn.get_attn_gradients()
            if magnitude:
                total_gradients += (gradients * gradients)
            else:
                total_gradients += gradients   
       
        W_state = (total_gradients / nsamples).clamp(min=0).mean(1).reshape(B, num_tokens, num_tokens)

        R = W_state * R.unsqueeze(0)
        print(f'SGradient R: {R.size()}')
        return R[:, 0, 1:]
    
    def SGradient_normAtt(self, input, index=None ,stdev_spread=.15, nsamples=3, magnitude=False):
        output = self.model(input, register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"top_class: {index[0]}") # 0 for forward, 1 for slow down, 2 for left, 3 for right

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        # num_tokens = w_featmap * h_featmap + 1 # +1 for the [CLS] token
        num_tokens = self.model.backbone.blocks[0].attn.get_attention_map().shape[-1]
        
        R = I = torch.eye(num_tokens, num_tokens).cuda() # initialize R^0 as the identity matrix
        for blk in self.model.backbone.blocks:
            att = blk.attn.get_attention_map()
            grad = blk.attn.get_attn_gradients()

            # weighted_norm = attn_weighted_norm(att, blk.attn.v, blk.attn.proj, blk.attn.dim, blk.attn.num_heads)
            weighted_norm = attn_weighted_norm(att, blk.attn.v, blk.attn.proj, self.dim, self.num_heads)

            cam = avg_heads(weighted_norm, grad) + I
            R = torch.matmul(cam, R)

        # smooth gradient     
        x = self.model.backbone.blocks[0].attn.get_input()
        B = x.shape[0] 
        # c, w, h = input.shape

        stdev = stdev_spread * (torch.max(input) - torch.min(input))
        total_gradients = torch.zeros((B, self.num_heads, num_tokens, num_tokens), dtype=torch.float32).cuda()

        for _ in range(nsamples):
            noise = torch.normal(0, stdev, input.shape) # create ramdom gaussian noise with mean 0 and std stdev
            x_plus_noise = input + noise

            # backward propagation
            output = self.model(x_plus_noise.cuda(), register_hook=True)
            one_hot = np.zeros((B, output.size()[-1]), dtype=np.float32)
            one_hot[np.arange(B), index] = 1
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * output)

            self.model.zero_grad()
            one_hot.backward(retain_graph=True)

            # cal Attention graident
            gradients = self.model.backbone.blocks[-1].attn.get_attn_gradients()
            if magnitude:
                total_gradients += (gradients * gradients)
            else:
                total_gradients += gradients   
       
        W_state = (total_gradients / nsamples).clamp(min=0).mean(1).reshape(B, num_tokens, num_tokens)

        R = W_state * R.unsqueeze(0) # element-wise multiplication
        print(f'SGradient_normAtt R: {R.size()}')
        return R[:, 0, 1:]
    


        

       
        

        
            
    

