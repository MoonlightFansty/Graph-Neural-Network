import torch
import torch.nn.functional as F
from Model.GNN import GCN_Net, GAT_Net, GraphSAGE_Net
from torch_geometric.datasets import Planetoid

epochs = 200
hidden = 16
learning_rate = 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GCN

# if __name__ == '__main__':
#     dataset = Planetoid(root='./tmp/Cora', name='Cora')
#     print((dataset[0].train_mask).sum())
#     print((dataset[0].test_mask).sum())
#     print((dataset[0].val_mask).sum())
#     print(dataset[0])
#
#     model = GCN_Net(dataset.num_node_features, hidden, dataset.num_classes).to(device)
#     data = dataset[0].to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
#     model.train()
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         out = model(data)
#         loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
#         loss.backward()
#         optimizer.step()
#
#     model.eval()
#     _, pred = model(data).max(dim=1)
#     correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum()
#     acc = int(correct) / int(data.test_mask.sum())
#     print('GCN:', acc)


# GraphSAGE

# if __name__ == '__main__':
#     dataset = Planetoid(root='./tmp/Cora', name='Cora')
#     print((dataset[0].train_mask).sum())
#     print((dataset[0].test_mask).sum())
#     print((dataset[0].val_mask).sum())
#     print(dataset[0])
#
#     model = GraphSAGE_Net(dataset.num_node_features, hidden, dataset.num_classes).to(device)
#     data = dataset[0].to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
#     model.train()
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         out = model(data)
#         loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
#         loss.backward()
#         optimizer.step()
#
#     model.eval()
#     _, pred = model(data).max(dim=1)
#     correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum()
#     acc = int(correct)/ int(data.test_mask.sum())
#     print('GraphSAGE',acc)


# GAT

if __name__ == '__main__':
    dataset = Planetoid(root='./tmp/Cora', name='Cora')
    print((dataset[0].train_mask).sum())
    print((dataset[0].test_mask).sum())
    print((dataset[0].val_mask).sum())
    print(dataset[0])

    model = GAT_Net(dataset.num_node_features, hidden, dataset.num_classes, heads=4).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    _, pred = model(data).max(dim=1)
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum()
    acc = int(correct)/ int(data.test_mask.sum())
    print('GAT',acc)