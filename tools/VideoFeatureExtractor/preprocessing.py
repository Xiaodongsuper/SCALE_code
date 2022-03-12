import torch as th

class Normalize(object):

    def __init__(self, mean, std):
        self.mean = th.FloatTensor(mean).view(1, 3, 1, 1)
        self.std = th.FloatTensor(std).view(1, 3, 1, 1)

    def __call__(self, tensor):
        tensor = (tensor - self.mean) / (self.std + 1e-8)
        return tensor

class Preprocessing(object):

    def __init__(self, type, FRAMERATE_DICT):
        self.type = type
        self.FRAMERATE_DICT = FRAMERATE_DICT
        if type == '2d':
            self.norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif type == '3d':
            self.norm = Normalize(mean=[110.6, 103.2, 96.3], std=[1.0, 1.0, 1.0])
        elif type == 's3dg':
            pass
        elif type == 'raw_data':
            pass

    def _zero_pad(self, tensor, size):
        n = size - len(tensor) % size
        if n == size:
            return tensor
        else:
            z = th.zeros(n, tensor.shape[1], tensor.shape[2], tensor.shape[3])
            return th.cat((tensor, z), 0)

    def __call__(self, tensor):
        if self.type == '2d':
            tensor = tensor / 255.0
            tensor = self.norm(tensor)
        elif self.type == '3d':
            tensor = self._zero_pad(tensor, 16)
            tensor = self.norm(tensor)
            tensor = tensor.view(-1, 16, 3, 112, 112)
            tensor = tensor.transpose(1, 2)
        elif self.type == 's3dg':
            tensor = tensor / 255.0
            tensor = self._zero_pad(tensor, self.FRAMERATE_DICT[self.type])
            # To Batch= T x 3 x H x W
            tensor_size = tensor.size()
            tensor = tensor.view(-1, self.FRAMERATE_DICT[self.type], 3, tensor_size[-2], tensor_size[-1])
            # To Batch x 3 x T x H x W
            tensor = tensor.transpose(1, 2)
        elif self.type == 'raw_data':
            tensor = tensor / 255.0
            tensor = self._zero_pad(tensor, self.FRAMERATE_DICT[self.type])
            # To Batch= T x 3 x H x W
            tensor_size = tensor.size()
            tensor = tensor.view(-1, self.FRAMERATE_DICT[self.type], 3, tensor_size[-2], tensor_size[-1])
            # To Batch x 3 x T x H x W
            tensor = tensor.transpose(1, 2)

        return tensor

