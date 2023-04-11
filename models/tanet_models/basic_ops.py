import torch


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.autograd.Function):

    def __init__(self, consensus_type, dim=1):
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None
    # @staticmethod
    def forward(self, input_tensor): #  input_tensor (bz, T, n_class)
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':   #  take the average of all frame-level prediction,   which is video-level prediction
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output
    # @staticmethod
    def backward(self, grad_output):
        if self.consensus_type == 'avg':
            grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
        elif self.consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in


class SegmentAvg_static(torch.autograd.Function):
    @staticmethod
    def forward(self, input_tensor, ):  #  input_tensor (bz, T, n_class)
        dim_ = 1
        self.save_for_backward(input_tensor)
        output = input_tensor.mean(dim=dim_, keepdim= True)
        return output
    @staticmethod
    def backward(self, grad_output ):
        dim_ = 1
        input_tensor, = self.saved_tensors
        shape_ = input_tensor.size()
        grad_in = grad_output.expand(shape_) / float(shape_[dim_])
        return grad_in

class SegmentIdentity_static(torch.autograd.Function):
    @staticmethod
    def forward(self, input_tensor, ):  #  input_tensor (bz, T, n_class)
        # dim_ = 1
        # self.save_for_backward(input_tensor)
        output = input_tensor
        return output
    @staticmethod
    def backward(self, grad_output ):
        # dim_ = 1
        # input_tensor, = self.saved_tensors
        # shape_ = input_tensor.size()
        grad_in = grad_output
        return grad_in




class ConsensusModule(torch.nn.Module):  # contains no parameters

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim
        assert self.dim == 1

    def forward(self, input): #  input_tensor (bz, T, n_class)

        if self.consensus_type == 'avg':
            return SegmentAvg_static.apply(input)
            # return input.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            return SegmentIdentity_static.apply(input)

        # return SegmentConsensus(self.consensus_type, self.dim)(input)