import sympy as sp
import numpy as np
import torch
import torch.nn as nn
import copy
from model._abstract_model import SequentialRecModel
from model._modules import FeedForward, LayerNorm

class TVRecModel(SequentialRecModel):
    def __init__(self, args):
        super(TVRecModel, self).__init__(args)

        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.reg_weight = args.reg_weight

        self.item_encoder = NVRecEncoder(args)
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False, inference=False):
        sequence_emb = self.get_sequence_emb(input_ids, pos_emb=False)
        pad_mask = (input_ids==0)
        
        item_encoded_layers = self.item_encoder([sequence_emb, pad_mask], 
                                                output_all_encoded_layers=True,
                                                inference=inference)
        
        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]

        return sequence_output

    def orthogonal_regularization(self, B, weight=1.0):
        # B: [m, K+1, 2]
        B_real = B[:, :, 0]  # [m, K+1]
        B_imag = B[:, :, 1]  # [m, K+1]

        def ortho_loss(matrix):
            # Normalize rows
            matrix = nn.functional.normalize(matrix, p=2, dim=1)
            # Compute (B B^T - I)
            gram = matrix @ matrix.T
            I = torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
            return ((gram - I) ** 2).sum()

        loss_real = ortho_loss(B_real)
        loss_imag = ortho_loss(B_imag)

        return weight * (loss_real + loss_imag)

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):    
        seq_output = self.forward(input_ids)
        seq_output = seq_output[:, -1, :]
        item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_output, item_emb.transpose(0, 1))
        loss = nn.CrossEntropyLoss()(logits, answers)
        
        ortho_reg = 0.0
        for block in self.item_encoder.blocks:
            basis = block.layer.basis  # [m, K+1, 2]
            ortho_reg += self.orthogonal_regularization(basis, weight=self.reg_weight)
            
        return loss + ortho_reg
    
class NVRecEncoder(nn.Module):
    def __init__(self, args):
        super(NVRecEncoder, self).__init__()
        self.args = args
        block = NVRecBlock(args)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])
        
    def forward(self, hidden_states, output_all_encoded_layers=False, inference=False):
        hidden_states, pad_mask = hidden_states
        all_encoder_layers = [ hidden_states ]

        for layer_module in self.blocks:
            hidden_states[pad_mask] = 0.0
            hidden_states = layer_module(hidden_states, inference)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        return all_encoder_layers

class NVRecBlock(nn.Module):
    def __init__(self, args):
        super(NVRecBlock, self).__init__()
        self.layer = NVRecLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, inference=False):
        layer_output = self.layer(hidden_states, inference)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output

class NVRecLayer(nn.Module):
    def __init__(self, args):
        super(NVRecLayer, self).__init__()
        self.args = args
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.seq_len = self.args.max_seq_length
        self.K = 50
        self.pad_seq_len = self.seq_len + self.K - 1
        
        if args.M > 0:
            self.basis = nn.Parameter(torch.randn(args.M, self.K+1, 2, dtype=torch.float32) * 1e-3)
            self.c_t = nn.Parameter(torch.randn(self.pad_seq_len, args.M, dtype=torch.float32) * 1e-3)
        else:
            self.basis = torch.stack([torch.eye(self.seq_len+1), torch.zeros(self.seq_len+1,self.seq_len+1)], dim=2).to(self.args.device)
            self.c_t = nn.Parameter(torch.randn(self.pad_seq_len, self.seq_len+1, dtype=torch.float32) * 1e-3)

        self.mask = torch.ones(self.pad_seq_len, self.seq_len, 2).to(self.args.device)
        self.mask[self.seq_len:] = 0

        self.func_H = None
        self.gft, self.igft, self.L = self.tranform_matrix()
            
    def adjacency_matrix(self):
        A = torch.zeros((self.pad_seq_len, self.pad_seq_len))
        A[0, -1] = 1
        A[1:, :-1].fill_diagonal_(1)
        return A.to(self.args.device)
    
    def tranform_matrix(self):
        A = self.adjacency_matrix()
        lmd, igft_matrix = torch.linalg.eig(A)

        L = torch.stack([lmd ** i for i in range(self.K+1)], dim=1)

        return torch.linalg.inv(igft_matrix), igft_matrix, L
    
    def pad_tensor(self, tensor):
        pad_item = torch.zeros_like(tensor[:, :self.K-1, :])
        return torch.cat([tensor, pad_item], axis=1)
    
    def calculate_func_H(self):
        B = torch.view_as_complex(self.basis)
        H = self.c_t.to(torch.complex64) @ (B/torch.norm(B, p=2, dim=1, keepdim=True))
        L = self.L
        F = self.igft * (H @ L.T)
        func_H = F @ self.gft
        self.func_H = torch.real(func_H)
        
    def forward(self, input_tensor, inference=False):
        # [batch, seq_len, hidden]
        _, seq_len, _ = input_tensor.shape
        padded_tensor = self.pad_tensor(input_tensor)

        if inference:
            x = self.func_H @ padded_tensor
            
        else:
            x_tilde = self.gft @ padded_tensor.to(torch.complex64)
            B = torch.view_as_complex(self.basis)
            H = self.c_t.to(torch.complex64) @ (B/torch.norm(B, p=2, dim=1, keepdim=True))
            L = self.L
            F = self.igft * (H @ L.T)
            x = F @ x_tilde
            x = torch.real(x)
        
        x = x[:, :seq_len, :]
        hidden_states = self.out_dropout(x) 
        hidden_states = hidden_states + input_tensor

        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states
