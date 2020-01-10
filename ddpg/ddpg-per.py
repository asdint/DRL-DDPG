import tensorflow as tf
import numpy as np
import gym
import os

#####################  hyper parameters  ####################

# Hyper Parameters
ENV_NAME = 'Pendulum-v0'
EPISODE = 200
STEP = 200
TEST = 5
LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
MEMORY_SIZE = 10000
steps = []
episodes = []
with_noise = False # True = with_noise; False = without_noise
RENDER = False
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
REPLACE_TARGET_FREQ = 2 # Update frequency of the target network

class OU_noise(object):
    def __init__(self, num_actions, action_low_bound= -2, action_high_bound= 2, dt= 0.001,
                 mu= 0.0, theta= 0.15, max_sigma= 0.2, min_sigma= 0.1):
        self.mu = mu  # 0.0
        self.theta = theta  # 0.15
        self.sigma = max_sigma  # 0.3
        self.max_sigma = max_sigma  # 0.3
        self.min_sigma = min_sigma  # 0.1
        self.dt = dt  # 0.001
        self.num_actions = num_actions  # 1
        self.action_low = action_low_bound  # -2
        self.action_high = action_high_bound  # 2
        self.reset()

    def reset(self):
        self.state = np.zeros(self.num_actions)

    # self.state = np.zeros(self.num_actions)
    def state_update(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.num_actions)  # np.random.randn()生成0,1的随机数
        self.state = x + dx

    def add_noise(self, action):
        self.state_update()
        state = self.state
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, self.dt)
        return np.clip(action + state, self.action_low, self.action_high)


class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = list(np.zeros(capacity, dtype=object))  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, transition):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = transition  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.b = np.array(self.data)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root

class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    epsilon = 0.001  # small amount to avoid zero priority
    alpha = 0.5  # [0~1] convert the importance of TD error to priority
    beta = 0.5  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.01
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.full_flag = False

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), \
                                     np.empty((n, self.tree.data[0].size)), \
                                     np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        if min_prob == 0:
            min_prob = 0.00001
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data

        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

###############################DDPG####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, train_dir="./ddpg_models", batch_size=32, MEMORY_SIZE=10000):
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.memory = Memory(capacity=MEMORY_SIZE)
        self.pointer = 0
        self.per_batch_size = batch_size
        self.learn_step = 0
        self.explore_noise = OU_noise(self.a_dim)
        self.sess = tf.Session()

        self.train_dir = train_dir
        if not os.path.isdir(self.train_dir):
            os.mkdir(self.train_dir)

        self.actor_lr = tf.placeholder(tf.float32, shape=[], name='actor_lr')
        self.critic_lr = tf.placeholder(tf.float32, shape=[], name='critic_lr')

        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params,
                                             self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        # td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.abs_errors = tf.reduce_sum(tf.abs(q_target - q), axis=1)  # for updating Sumtree
        self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(q_target, q))
        self.ctrain = tf.train.AdamOptimizer(self.critic_lr).minimize(self.loss, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(self.actor_lr).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())
    def choose_action(self, s, with_noise):
        action = self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
        if with_noise:
            noise = self.explore_noise.add_noise(action)
            action = action + noise
        return action

    def learn(self, actor_lr_input, critic_lr_input, per_flag=True):

        if per_flag:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.per_batch_size) # sample for learning
            batch_states = batch_memory[:,0:3]
            batch_actions = batch_memory[:,3:4]
            batch_rewards = [data[4] for data in batch_memory]
            batch_states_ = batch_memory[:,5:8]

            bs = np.array(batch_states)
            ba = np.array(batch_actions)
            br = np.array(batch_rewards)
            bs_ = np.array(batch_states_)
            br = br[:, np.newaxis] # Move the original (n,) to the row and add a new column

            self.sess.run(self.atrain, {self.S: bs, self.actor_lr: actor_lr_input})
            _, abs_errors, cost = self.sess.run([self.ctrain, self.abs_errors, self.loss],
                                                {self.S: bs, self.a: ba, self.R: br, self.S_: bs_,
                                                 self.critic_lr: critic_lr_input,
                                                 self.ISWeights: ISWeights})

            self.memory.batch_update(tree_idx, abs_errors)  # update priority

        self.learn_step += 1

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        self.memory.store(transition)
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            # new_actor_layer = tf.layers.dense(net, 20, activation=tf.nn.relu, name='new_actor_layer', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            # new_critic_layer = tf.layers.dense(net, 300, activation=tf.nn.relu, name='new_critic_layer',
            #                                    trainable=trainable)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

    def update_target_q_network(self, episode):
        # update target Q netowrk by soft_replace
        if episode % REPLACE_TARGET_FREQ == 0:
            self.sess.run(self.soft_replace)
            # print('episode '+str(episode) +', target Q network params replaced!')

    def load_network(self, saver, load_path):
        checkpoint = tf.train.get_checkpoint_state(load_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            # self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            saver.restore(self.sess, tf.train.latest_checkpoint(load_path))
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
            self.learn_step = int(checkpoint.model_checkpoint_path.split('-')[-1])
        else:
            print("Could not find old network weights")

    def save_network(self, time_step, saver, save_path):
        saver.save(self.sess, save_path + 'network', global_step=time_step,
                   write_meta_graph=False)

###############################  training  ####################################

def main():
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    agent = DDPG(a_dim, s_dim, a_bound)
    total_steps = 0
    var = 3
    for episode in range(EPISODE):
        state = env.reset()
        ep_reward = 0
        # train
        for step in range(STEP):
            if RENDER:
                env.render()
            action = agent.choose_action(state, with_noise)
            action = np.clip(np.random.normal(action, var), -2, 2)
            next_state, reward, done, _ = env.step(action)

            agent.store_transition(state,action,reward/10,next_state)

            if agent.pointer > MEMORY_SIZE:
                var *= .9995  # decay the action randomness
                if episode >= 50:
                    yy = 0
                agent.learn(LR_A, LR_C, per_flag=True)

            state = next_state
            ep_reward += reward
            total_steps += 1
            if done:
                print('episode ', episode, ' finished')
                steps.append(total_steps)
                episodes.append(episode)
                break
            if episode % 1 == 0:
                if step == STEP - 1:
                    print('Episode:', episode, ' Reward: %i' % int(ep_reward))
                    break

        # Test every 100 episodes
        if episode != 0 and episode % 100 ==0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    if RENDER:
                        env.render()
                    action = agent.choose_action(state, False)
                    state,reward,done,_ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('episode: ',episode,'Evaluation Average Reward:',ave_reward)
        agent.update_target_q_network(episode)

if __name__ == '__main__':
    main()