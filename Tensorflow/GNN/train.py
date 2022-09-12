import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long) # graph connectivity [2, num_edges]
x = torch.tensor([[-1], [0], [1]], dtype=torch.float) # node feature [num_nodes, num_node_features]

data = Data(x=x, edge_index=edge_index)

print(data)
print(data.keys)
print(data['x'])

