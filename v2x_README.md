# V2X Marl Readme

我们从程序入口，将程序运行流过一遍。

main.py、run.py的run方法都一样，主要进行的是配置解析以及实验环境初始化工作，此处跳过。

## 一、算法主体循环

run.py的run_sequential方法开始不同

```python
def run_sequential(args, logger):
```

1. 选择runner

   ```python
   runner = r_REGISTRY[args.runner](args=args, logger=logger)
   ```

   我们需要在配置文件中指定runner为我们重写的runner，目前为v2xepisode，对应v2x_episode_runner.py文件中的类。

2. 获取环境信息

   ```python
   env_info = runner.get_env_info()
   args.n_agents = env_info["n_agents"]
   args.n_actions = env_info["n_actions"]
   args.state_shape = env_info["state_shape"]
   ```

   这说明必须存在以上三个信息。

3. 准备scheme、groups、preprocess，用于告知buffer自己的存储任务、以及mac的控制任务

   ```python
   scheme = {
       "state": {"vshape": env_info["state_shape"]},
       "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
       "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
       "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
       "reward": {"vshape": (1,)},
       "terminated": {"vshape": (1,), "dtype": th.uint8},
       "alive_allies": {"vshape": (env_info["n_agents"],)},
   }
   groups = {
       "agents": args.n_agents
   }
   preprocess = {
       "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
   }
   ```

   scheme的的格式如上。我们的原始算法中并不需要在buffer中保存这么多类型的数据，我们只需要state、action、reward即可。但是我们没有必要完全复刻算法，因为原算法效果很差，我们完全可以在此基础上增加一些信息。此处对scheme标识存储的数据做完整说明。

   1. state：似乎是全局状态，但是在原项目以及v2x环境中都没有被应用，所以暂时不表。
   2. obs：所有agent的本地观测，即它的“视角”
   3. actions：每个agent采取的动作
   4. avail_actions：每个agent可以采取的动作，似乎可采取的动作是1，其他的是0？在目前的离散v2x问题中，该部分意义不大。
   5. reward：所有agent共享的全局奖励
   6. terminated：标志episode是否结束。在v2x问题中，因为每个episode的时间步数固定，所以没有意义。
   7. alive_allies：标志还有哪些友军存活。在v2x问题中获取可以用于表明哪些agent的v2v还未完成传输？

4. 创建buffer、创建mac

   需要在配置文件中指定mac为我们重写的mac，因为算法不使用RNN而是使用多层dense也就是MLP来作为网络主体。

5. runner.setup

   ```python
   runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
   ```

   该方法主要是为runner指定scheme，而runner使用scheme的目的是确定其EpisodeBatch保存的数据格式。scheme显然应与ReplayBuffer的一致。

6. 创建learner

   ```python
   learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
   ```

   learner是与算法直接挂钩的，所以需要重写一个learner。此处需要参考已有learner，也可能可以直接使用，**待定**。

   learner用于更新模型，实现反向传播。主要和使用的模型有关。

7. 对于每个episode，执行如下操作。

   1. 运行一个episode，获取一个batch的数据，并将其加入buffer

      ```python
      episode_batch = runner.run(test_mode=False)
      buffer.insert_episode_batch(episode_batch)
      ```

      对于非并行runner，该batch的batch_size为1。并行runner则根据同时运行的线程数决定。

   2. 当样本量足够时，进行抽样，

      ```python
      episode_sample = buffer.sample(args.batch_size)
      max_ep_t = episode_sample.max_t_filled()
      episode_sample = episode_sample[:, :max_ep_t]
      ```

      计算训练数据批次中最大填充时间步数，将训练数据批次截断到最大填充时间步数

   3. 利用sample的batch数据训练模型

      ```python
      learner.train(episode_sample, runner.t_env, episode)
      ```

   4. 做每个episode的test。

      在原v2x算法中没有这部分，待定。大概是validation？先跳过。

   5. 保存模型

   6. 打印数据

​    

综上，共有以下几个需要重点关注的部分：

1. **mac层实现方法**

   我们不使用RNN作为agent主体，而采用mlp。不过后续可以考虑使用RNN来保留一个episode内的时序关系。

   已完成

2. runner的设计

   主要接口是run方法，其余方法或作为辅助或在其他函数中有调用。等待进一步探索。

   已完成

3. **learner的设计**

   train方法设计。



## 二、 一个Episode内如何运行？

我们先理解先有的代码，再通过对比它与v2x marl的区别，进行修改。此处将会递归地分析同属于EpisodeRunner的的方法，但不会展开过于细节的内容。

```python
def run(self, test_mode=False):
```

1. 重置runner

   ```python
   self.reset()
   terminated = False # 标记episode是否结束
   episode_return = 0 # episode累计奖励值
   ```

   具体来说，该函数重新创建了一个batch用于保存数据，刷新了环境，并把t归零

   ```python
   def reset(self):
       self.batch = self.new_batch()
       self.env.reset()
       self.t = 0
   ```

2. 初始化mac的隐藏层。

   ```python
   self.mac.init_hidden(batch_size=self.batch_size)
   ```

   我觉得这里写的比较奇怪。还不太理解。

3. 进入episode循环：

   1. 将先前状态等信息存入

      ```python
      pre_transition_data = {
          "state": [self.env.get_state()],
          "avail_actions": [self.env.get_avail_actions()],
          "obs": [self.env.get_obs()],
          "alive_allies": self.env.get_alive_agents(),
      }
      
      self.batch.update(pre_transition_data, ts=self.t)
      ```

      

   2. 将batch输入给mac获得动作，将动作输入环境，得到奖励和环境信息。

      ```python
      actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
      
      reward, terminated, env_info = self.env.step(actions[0])
      episode_return += reward
      ```

      

   3. 存储行动后数据

      ```python
      post_transition_data = {
          "actions": actions,
          "reward": [(reward,)],
          "terminated": [(terminated != env_info.get("episode_limit", False),)],
      }
      self.batch.update(post_transition_data, ts=self.t)
      
      ```

      此处需要特别说明几点。第一，这个runner适用的环境是一个episode内各时间步相关度非常高的环境，例如游戏。于是在选择动作时直接将batch作为输入了。第二，这个算法没有存储末状态，因为始终使用连续的episode内数据训练，但我们的环境每个时间步后环境会变化（fastfading），所以必须要存储末状态。

      如果要用于v2x，至少需要做如下修改：

      1. 改变transition的存储方式，必须存储末状态。
      2. 改变select_action的处理，要么不输入batch数据而改为只输入pre_transition_data，要么在函数内挑选batch数据。我选择后者。

 

## 三、mac控制器是如何工作的？

由于第二节中，主要的调用的是来自mac的方法，并且在第一节中，mac也是没有阐释的部分，因此将其mac展开为一个章节。

在run.py中初始化了mac，即全局的multi-agent-controller，其接受经典的三件套作为输入。

```python
mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
```

### 初始化函数

我们首先分析初始化函数，对于其中尚不能理解的部分，等待后续方法实现部分解释。

```python
def __init__(self, scheme, groups, args):
    self.n_agents = args.n_agents
    self.args = args
    input_shape = self._get_input_shape(scheme)
    self._build_agents(input_shape)
    if self.args.input == 'hidden':
        self._build_consensus_builder(self.args.rnn_hidden_dim)
    elif self.args.input == 'obs':
        self._build_consensus_builder(input_shape)
    self._build_embedding_net()
    self.agent_output_type = args.agent_output_type

    self.action_selector = action_REGISTRY[args.action_selector](args)

    self.hidden_states = None
    self.obs_center = th.zeros(1, self.args.consensus_builder_dim).cuda()
```

1. 利用scheme获取输入的向量维度，除此以外scheme别无他用

   ```python
   input_shape = self._get_input_shape(scheme)
   ```

2. 根据输入shape为每个agent构建网络

   ```python
   self._build_agents(input_shape)
   ```
   
3. 一方面用于区分是否使用consensus_builder，以及使用方法，但是没太懂。

   ```python
   if self.args.input == 'hidden':
       self._build_consensus_builder(self.args.rnn_hidden_dim)
   elif self.args.input == 'obs':
       self._build_consensus_builder(input_shape)
   ```

4. 初始化action_selector，后续将展开说明。

   ```python
   self.action_selector = action_REGISTRY[args.action_selector](args)
   ```

5. 用于——

   ```python
   self.hidden_states = None
   ```

### 动作选择

```python
def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
    # Only select actions for the selected batch elements in bs
    avail_actions = ep_batch["avail_actions"][:, t_ep]
    agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
    chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,test_mode=test_mode)
    return chosen_actions
```

1. 获取可用动作，在我们的问题里没有这个字段
2. 利用forward获取网络输出的动作
3. 再通过action_selector来根据某种策略选择动作。例如epsilon-greedy

### 前向传播

```python
def forward(self, ep_batch, t, test_mode=False):
```

主要代码用于利用consensus计算共识了，我们基本不太需要。但在此也分析一遍。

1. 构建真正的神经网络输入

   ```python
   agent_inputs = self._build_inputs(ep_batch, t)
   ```

   主要工作就是把需要的数据从batch中取出来，然后添加一些数据，再展平。

2. rnn的机制与共识机制
   ```python
   self.hidden_states = self.agent.calc_hidden(agent_inputs, self.hidden_states)
   with th.no_grad():
       if self.args.input == 'hidden':
           latent_state = self.consensus_builder.calc_student(self.hidden_states)
       elif self.args.input == 'obs':
           latent_state = self.consensus_builder.calc_student(agent_inputs)
   
       # latent_state = latent_state - latent_state.max(-1, keepdim=True)[0].detach()
       latent_state_id = F.softmax(latent_state, dim=-1).detach().max(-1)[1].unsqueeze(-1)
       latent_state_id[
           ep_batch['alive_allies'][:, t].reshape(*latent_state_id.size()) == 0] = self.args.consensus_builder_dim
       latent_state_embedding = self.embedding_net(latent_state_id.squeeze(-1))
   agent_outs = self.agent.calc_value(latent_state_embedding, self.hidden_states)
   ```
   hidden_states是rnn的中间输出。通过保存该层输出，实现环境和RNN的一同迭代。

   剩余部分主要处理了结合consensus的代码。我们使用consensus时大概率要进行修改。
   
3. 对于不可用动作进行mask

   ```python
   # Softmax the agent outputs if they're policy logits
   if self.agent_output_type == "pi_logits":
   
       if getattr(self.args, "mask_before_softmax", True):
           # Make the logits for unavailable actions very negative to minimise their affect on the softmax
           reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
           agent_outs[reshaped_avail_actions == 0] = -1e10
   
       agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
       if not test_mode:
           # Epsilon floor
           epsilon_action_num = agent_outs.size(-1)
           if getattr(self.args, "mask_before_softmax", True):
               # With probability epsilon, we will pick an available action uniformly
               epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()
   
           agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                         + th.ones_like(agent_outs) * self.action_selector.epsilon / epsilon_action_num)
   
           if getattr(self.args, "mask_before_softmax", True):
               # Zero out the unavailable actions
               agent_outs[reshaped_avail_actions == 0] = 0.0
   ```

### 构建agent网络

该代码在多agent之间进行了参数共享，实际上只用了一个网络。那么是如何一次性取得多个agent的动作的呢？

通过_build_inputs函数并结合episode_batch的结构可以得到答案

```python
def _build_inputs(self, batch, t):
    # Assumes homogenous agents with flat observations.
    # Other MACs might want to e.g. delegate building inputs to each agent
    bs = batch.batch_size
    inputs = []
    inputs.append(batch["obs"][:, t])  # b1av
    if self.args.obs_last_action:
        if t <= 0:
            inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
        else:
            inputs.append(batch["actions_onehot"][:, t-1])
    if self.args.obs_agent_id:
        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

    inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
    return inputs
```

此处：

```python
inputs.append(batch["obs"][:, t])  # b1av
```

这一行最关键的一点是，加入到inputs的数据结构是一个形状为[n_agents, ...]的张量。这首先意味着每个agent的数据是区分开的，即：$inputs[0][i]$表示的是i号agent的obs数据。

最后，

```python
inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
```

reshape的目的是将batch引入的维度和n_agents引入的维度融合，并将剩余的维度展平。然后通过cat在dim=1的维度上拼接从而得到一个二维数组，$\text{shape=(bs*self.n_agents, input_shape)}$。

当这样的数组输入给网络时，第一维会被解释为batch，也就是等价于网络对inputs[0], inputs[1]...依次进行计算。从而利用网络对batch的支持，实现n个agent的运算（当然如果bs不等于1，后面为了拆分出每个batch的输入肯定还要再运算的）。



### 四、环境需要实现什么？

