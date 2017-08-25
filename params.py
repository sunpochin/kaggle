from model.u_net import get_unet_128, get_unet_256, get_unet_512, get_unet_1024
from model.u_net import get_convnet_simple, get_unet_1918_1280

# use larger downsample for faster modeling, but not accurate.
downsample = 1
#input_size = 128
#input_size = 1024
# 0 meaning NO resize.
input_size = 0

max_epochs = 50
#batch_size = 16
batch_size = 3

orig_width = 1918
orig_height = 1280

threshold = 0.5

#model = get_unet_128()
#model = get_convnet_simple()
#model = get_unet_1024()
model = get_unet_1918_1280()


