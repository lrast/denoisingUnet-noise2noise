# Learning to remove noise from images

This is a reproduction of, and follow-up on, [Lehtinen et al. (2018) Noise2Noise: Learning Image Restoration without Clean Data], which trained neural networks to denoise images without access to ground truth data. The key idea was to train the network to, given a noisy images as input, try to reproduce another noisy image of the same underlying image. By training across all noisy image pairs, the network was forced to learn a middle ground output between the noisy images. This middle ground depended on the loss function that the network was trained with: mean for L2, median for L1, etc.

The paper reported superior results in some cases with this paired noise approach compared to learning on the mean image, or even the ground truth image. It is this result that I would like to dig into and understand more depth.




