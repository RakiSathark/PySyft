import syft
from syft.frameworks.torch.tensors.abstract import AbstractTensor


class LogTensor(AbstractTensor):
    def __init__(self, parent: AbstractTensor = None, owner=None, id=None):
        """Initializes a PointerTensor.

        Args:
            parent: An optional AbstractTensor wrapper around the PointerTensor
                which makes it so that you can pass this PointerTensor to all
                the other methods/functions that PyTorch likes to use, although
                it can also be other tensors which extend AbstractTensor, such
                as custom tensors for Secure Multi-Party Computation or
                Federated Learning.
            owner: An optional BaseWorker object to specify the worker on which
                the pointer is located. It is also where the pointer is
                registered if register is set to True. Note that this is
                different from the location parameter that specifies where the
                pointer points to.
            id: An optional string or integer id of the PointerTensor.
        """
        self.parent = parent
        self.owner = owner
        self.id = id
        self.child = None

    def __str__(self) -> str:
        if hasattr(self, "child"):
            return type(self).__name__ + ">" + self.child.__str__()
        else:
            return type(self).__name__

    def __repr__(self) -> str:
        if hasattr(self, "child"):
            return type(self).__name__ + ">" + self.child.__repr__()
        else:
            return type(self).__name__

    def on(self, tensor):
        self.child = tensor
        return self

    @classmethod
    def handle_method_command(cls, command):
        cmd, self, args = command  # TODO: add kwargs

        # Do what you have to
        print("Logtensor logging", cmd)

        new_self, new_args = syft.frameworks.torch.hook_args.hook_method_args(
            cmd, self, args
        )  # TODO add new_kwargs, kwargs

        new_command = (cmd, new_self, new_args)

        response = type(new_self).handle_method_command(new_command)

        response, _ = syft.frameworks.torch.hook_args.hook_method_response(
            cmd, (response, 1), wrap_type=cls
        )

        return response
