# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F

# # class GATLayer(nn.Module):
# #     def __init__(self, in_features, out_features):
# #         super(GATLayer, self).__init__()
# #         self.W = nn.Linear(in_features, out_features, bias=False)
# #         self.a = nn.Linear(2 * out_features, 1, bias=False)

# #     def forward(self, h, adj):
# #         """
# #         h: (batch, N, in_features)
# #         adj: (N, N)
# #         returns: (batch, N, out_features)
# #         """
# #         batch_size, N, _ = h.size()
# #         Wh = self.W(h)  # (batch, N, out_features)

# #         # Attention mechanism
# #         a_input = torch.cat([
# #             Wh.unsqueeze(2).repeat(1,1,N,1),
# #             Wh.unsqueeze(1).repeat(1,N,1,1)
# #         ], dim=-1)  # (batch, N, N, 2*out_features)
# #         e = F.leaky_relu(self.a(a_input).squeeze(-1))  # (batch, N, N)

# #         adj_batch = adj.unsqueeze(0).expand(batch_size, -1, -1)
# #         zero_vec = -9e15 * torch.ones_like(e)
# #         attention = torch.where(adj_batch>0, e, zero_vec)
# #         attention = F.softmax(attention, dim=-1)

# #         h_prime = torch.bmm(attention, Wh)  # (batch, N, out_features)
# #         return h_prime

# # class STGAAT(nn.Module):
# #     def __init__(self, in_features, hidden_dim=64, out_dim=1, num_nodes=16, seq_len=12, dropout=0.2):
# #         super(STGAAT, self).__init__()
# #         self.num_nodes = num_nodes
# #         self.seq_len = seq_len
# #         self.dropout = nn.Dropout(dropout)
# #         self.gat = GATLayer(in_features, hidden_dim)
# #         self.fc = nn.Linear(hidden_dim * seq_len, out_dim)

# #     def forward(self, x, adj):
# #         """
# #         x: (batch, seq_len, N, in_features)
# #         adj: (N, N)
# #         returns: (batch, N) if out_dim=1
# #         """
# #         batch_size, seq_len, N, in_feat = x.size()
# #         outs = []

# #         for t in range(seq_len):
# #             h = self.gat(x[:, t, :, :], adj)  # (batch, N, hidden_dim)
# #             if self.training:
# #                 h = h + 0.01 * torch.randn_like(h)  # Gaussian noise
# #             h = self.dropout(F.relu(h))
# #             outs.append(h)

# #         h = torch.cat(outs, dim=-1)  # (batch, N, hidden_dim*seq_len)
# #         out = self.fc(h)             # (batch, N, out_dim)
# #         if out.shape[-1] == 1:
# #             out = out.squeeze(-1)    # (batch, N)
# #         return out


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class GATLayer(nn.Module):
#     def __init__(self, in_features, out_features):
#         super().__init__()
#         self.W = nn.Linear(in_features, out_features, bias=False)
#         self.a = nn.Linear(2*out_features, 1, bias=False)

#     def forward(self, h, adj):
#         # h: (batch, N, in_features)
#         Wh = self.W(h)  # (batch, N, out_features)
#         N = Wh.size(1)
#         a_input = torch.cat([Wh.unsqueeze(2).repeat(1,1,N,1),
#                              Wh.unsqueeze(1).repeat(1,N,1,1)], dim=-1)  # (batch,N,N,2*out_features)
#         e = F.leaky_relu(self.a(a_input).squeeze(-1))  # (batch,N,N)
#         zero_vec = -9e15*torch.ones_like(e)
#         attention = torch.where(adj.unsqueeze(0) > 0, e, zero_vec)
#         attention = F.softmax(attention, dim=-1)
#         h_prime = torch.matmul(attention, Wh)  # (batch,N,out_features)
#         return h_prime

# class STGAAT(nn.Module):
#     def __init__(self, num_nodes, in_features=2, gat_out=32, lstm_hidden=64, n_pred=1):
#         super().__init__()
#         self.num_nodes = num_nodes
#         self.in_features = in_features
#         self.gat_out = gat_out
#         self.lstm_hidden = lstm_hidden
#         self.n_pred = n_pred

#         self.gat = GATLayer(in_features, gat_out)
#         self.dropout = nn.Dropout(0.2)
#         self.lstm = nn.LSTM(gat_out, lstm_hidden, batch_first=True)
#         self.fc = nn.Linear(lstm_hidden, n_pred)

#     def forward(self, x, adj):
#         # x: (batch, seq_len, N, in_features)
#         if self.training:
#             x = x + 0.01 * torch.randn_like(x)

#         batch, seq_len, N, F_in = x.size()
#         outs = []
#         for t in range(seq_len):
#             h = self.gat(x[:,t,:,:], adj)  # (batch,N,gat_out)
#             h = F.relu(self.dropout(h))
#             outs.append(h.unsqueeze(1))
#         h_seq = torch.cat(outs, dim=1)  # (batch,seq_len,N,gat_out)
#         h_seq = h_seq.permute(0,2,1,3).reshape(batch*N, seq_len, self.gat_out)
#         lstm_out,_ = self.lstm(h_seq)  # (batch*N, seq_len, lstm_hidden)
#         out = self.fc(lstm_out[:,-1,:]) # (batch*N, n_pred)
#         out = out.view(batch, N, self.n_pred)
#         if self.n_pred==1:
#             out = out.squeeze(-1)  # (batch,N)
#         return out



import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GATLayer, self).__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)

    def forward(self, h, adj):
        # h: (batch, N, in_features)
        Wh = self.W(h)  # (batch, N, out_features)
        N = Wh.size(1)

        a_input = torch.cat([Wh.unsqueeze(2).repeat(1,1,N,1),
                             Wh.unsqueeze(1).repeat(1,N,1,1)], dim=-1)  # (batch, N, N, 2*out_features)
        e = F.leaky_relu(self.a(a_input).squeeze(-1))  # (batch, N, N)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.unsqueeze(0) > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)  # (batch, N, N)
        h_prime = torch.bmm(attention, Wh)  # (batch, N, out_features)
        return h_prime


class STGAAT(nn.Module):
    def __init__(self, num_nodes, in_features=2, gat_out=32, lstm_hidden=64, n_pred=1):
        super(STGAAT, self).__init__()
        self.num_nodes = num_nodes
        self.gat = GATLayer(in_features, gat_out)
        self.lstm = nn.LSTM(gat_out, lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, n_pred)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, adj):
        # x: (batch, seq_len, N, in_features)
        if self.training:
            x = x + 0.01 * torch.randn_like(x)  # Gaussian noise for robustness

        batch, seq_len, N, _ = x.shape
        gat_outs = []
        for t in range(seq_len):
            h = self.gat(x[:, t, :, :], adj)  # (batch, N, gat_out)
            h = self.dropout(F.relu(h))
            gat_outs.append(h)

        h_seq = torch.stack(gat_outs, dim=1)  # (batch, seq_len, N, gat_out)
        # Merge nodes dimension for LSTM: treat each node separately
        h_seq = h_seq.permute(0,2,1,3)        # (batch, N, seq_len, gat_out)
        h_seq = h_seq.reshape(batch*N, seq_len, -1)  # (batch*N, seq_len, gat_out)

        lstm_out, _ = self.lstm(h_seq)        # (batch*N, seq_len, lstm_hidden)
        last_hidden = lstm_out[:,-1,:]        # (batch*N, lstm_hidden)
        out = self.fc(last_hidden)            # (batch*N, n_pred)
        out = out.view(batch, N, -1)          # (batch, N, n_pred)
        return out.squeeze(-1)                # (batch, N) if n_pred=1
