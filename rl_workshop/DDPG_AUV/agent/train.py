import torch.nn.functional as F

BATCH_SIZE = 32
GAMMA = 0.98
TAU = 0.99

def train(actor, actor_target, critic, critic_target, memory, critic_optimizer, actor_optimizer):
    s, a, r, s_prime, done_mask = memory.sample(BATCH_SIZE)

    target = r + GAMMA * critic_target(s_prime, actor_target(s_prime)) * done_mask
    critic_loss = F.smooth_l1_loss(critic(s, a), target.detach())
    actor_loss = -critic(s, actor(s)).mean()

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()


def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - TAU) + param.data * TAU)