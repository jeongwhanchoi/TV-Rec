import torch
import torch.nn as nn
from model._modules import LayerNorm

class SequentialRecModel(nn.Module):
    def __init__(self, args):
        super(SequentialRecModel, self).__init__()
        self.args = args
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.batch_size = args.batch_size

    def get_sequence_emb(self, sequence, pos_emb=True):
        item_embeddings = self.item_embeddings(sequence)
        if pos_emb:
            seq_length = sequence.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
            position_ids = position_ids.unsqueeze(0).expand_as(sequence)
            position_embeddings = self.position_embeddings(position_ids)
            sequence_emb = item_embeddings + position_embeddings
        else:
            sequence_emb = item_embeddings
            self.position_embeddings = None
        
        sequence_emb = self.dropout(self.LayerNorm(sequence_emb))

        return sequence_emb

    def init_weights(self, module):
        """ Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, all_sequence_output=False, inference=False):
        pass

    def predict(self, input_ids, user_ids, all_sequence_output=False):
        return self.forward(input_ids, user_ids, all_sequence_output, inference=True)

    def calculate_loss(self, input_ids, answers):
        pass

