# Third-party components

The MIT license in this repository applies to its training wrappers and supporting scripts. Dependencies, datasets and downloaded weights retain their own licenses and terms. No dependency wheel, pretrained weight, private dataset or trained checkpoint is included in the source release.

- [LeRobot](https://github.com/huggingface/lerobot): ACT implementation and dataset/processor/training APIs; Apache-2.0.
- [PyTorch](https://github.com/pytorch/pytorch) and [torchvision](https://github.com/pytorch/vision): model execution and ResNet18 backbone; see their license files.
- [TorchCodec](https://github.com/meta-pytorch/torchcodec) and [FFmpeg](https://ffmpeg.org/legal.html): video decoding; FFmpeg terms depend on its build.
- [Matplotlib](https://matplotlib.org/stable/project/license.html): training plots.
- ResNet18 ImageNet weights are downloaded from the official PyTorch distribution into the local cache, with checksum validation. They are not redistributed in this source archive.

Users supply their own datasets and are responsible for permission to use or redistribute them.
