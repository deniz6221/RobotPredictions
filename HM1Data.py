import numpy as np
from homework1 import Hw1Env
import torch

N = 100

positions = torch.zeros(N, 2, dtype=torch.float)
actions = torch.zeros(N, dtype=torch.uint8)
before_imgs = torch.zeros(N, 3, 128, 128, dtype=torch.uint8)
after_imgs = torch.zeros(N, 3, 128, 128, dtype=torch.uint8)

env = Hw1Env(render_mode="offscreen")
for i in range(N):
    env.reset()
    action_id = np.random.randint(4)
    _, img_before = env.state()
    env.step(action_id)
    pos_after, img_after = env.state()

    positions[i] = torch.tensor(pos_after)
    actions[i] = action_id
    before_imgs[i] = img_before
    after_imgs[i] = img_after
    
    env.reset()


torch.save(positions, "./test_data/positions.pt")    
torch.save(actions, "./test_data/actions.pt")    
torch.save(before_imgs, "./test_data/before_imgs.pt")    
torch.save(after_imgs, "./test_data/after_imgs.pt")    