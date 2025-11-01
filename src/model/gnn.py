import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1):
        super(GraphTransformer, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(TransformerConv(in_channels=in_channels,
                                          out_channels=hidden_channels//num_heads,
                                          heads=num_heads,
                                          edge_dim=in_channels,
                                          dropout=dropout))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(in_channels=hidden_channels,
                                              out_channels=hidden_channels//num_heads,
                                              heads=num_heads, edge_dim=in_channels,
                                              dropout=dropout))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(TransformerConv(in_channels=hidden_channels,
                                          out_channels=out_channels//num_heads,
                                          heads=num_heads,
                                          edge_dim=in_channels,
                                          dropout=dropout))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=adj_t, edge_attr=edge_attr)
            x = self.bns[i](x)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index=adj_t, edge_attr=edge_attr)
        return x, edge_attr


load_gnn_model = {
    'gt': GraphTransformer,
}
