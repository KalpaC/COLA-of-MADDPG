# Read Me For Consensus Builder

该篇内容用于梳理共识是如何构建的，并理清其中有哪些trick和它们的作用。



ConsensusBuilder:

```python
# * student
self.view_obs_net = MLP(self.obs_dim, self.consensus_builder_hidden_dim * 2, self.consensus_builder_hidden_dim)
self.project_net = MLP(self.consensus_builder_hidden_dim, self.consensus_builder_hidden_dim, self.consensus_builder_dim)

# * teacher
self.teacher_view_obs_net = MLP(self.obs_dim, self.consensus_builder_hidden_dim * 2, self.consensus_builder_hidden_dim)
self.teacher_project_net = MLP(self.consensus_builder_hidden_dim, self.consensus_builder_hidden_dim, self.consensus_builder_dim)
```

学生与老师各具有一个主网络和目标网络，然后利用update()同步参数，没啥特别好说的，主要是梳理用法。

网络结构：

![img](README_for_CB.assets/image.png)

网络将CB和环境共同作为输入，CB以环境作为输入，生成onehot向量，再经过嵌入层，最后共同输入给网络。

所以实际上引入了两个网络：Consensus Builder以及Embedding Net