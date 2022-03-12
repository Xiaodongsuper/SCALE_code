import sys
import torch as th
import torchvision.models as models
from videocnn.models import resnext, s3dg
from torch import nn


class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return th.mean(x, dim=[-2, -1])

def init_weight(model, state_dict, should_omit="s3dg."):
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if 'gamma' in key:
            new_key = key.replace('gamma', 'weight')
        if 'beta' in key:
            new_key = key.replace('beta', 'bias')
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    if should_omit is not None:
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            if key.find(should_omit) == 0:
                old_keys.append(key)
                new_key = key[len(should_omit):]
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='')

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(model.__class__.__name__, "\n   " + "\n   ".join(missing_keys)))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(model.__class__.__name__, "\n   " + "\n   ".join(unexpected_keys)))
    if len(error_msgs) > 0:
        print("Weights from pretrained model cause errors in {}: {}".format(model.__class__.__name__, "\n   " + "\n   ".join(error_msgs)))

    return model

def get_model(args):
    assert args.type in ['2d', '3d', 's3dg']
    if args.type == '2d':
        print('Loading 2D-ResNet-152 ...')
        model = models.resnet152(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-2], GlobalAvgPool())
        model = model.cuda()
    elif args.type == '3d':
        print('Loading 3D-ResneXt-101 ...')
        model = resnext.resnet101(
            num_classes=400,
            shortcut_type='B',
            cardinality=32,
            sample_size=112,
            sample_duration=16,
            last_fc=False)
        model = model.cuda()
        model_data = th.load(args.resnext101_model_path)
        model.load_state_dict(model_data)
    elif args.type == 's3dg':
        print('Loading S3DG ...')
        model = s3dg.S3D(last_fc=False)
        model = model.cuda()
        model_data = th.load(args.s3d_model_path)
        model = init_weight(model, model_data)


    model.eval()
    print('loaded')
    return model

if __name__ == "__main__":
    model = resnext.resnet101(
        num_classes=400,
        shortcut_type='B',
        cardinality=32,
        sample_size=112,
        sample_duration=16,
        last_fc=False)
    print(model)
