import torch
print(torch.hub.get_dir())  # /home/user/.cache/torch/hub
# (base) user@d36fbb52e6cd:/workdir$ ls /home/user/.cache
# matplotlib  pip
# (base) user@d36fbb52e6cd:/workdir$ ls -la ~
# drwxr-xr-x 1 user user  4096 Oct 17 02:27 .cache
# chmod -R 777 ~/.cache/

import torchvision
print(torchvision.models.resnet101(pretrained=False))