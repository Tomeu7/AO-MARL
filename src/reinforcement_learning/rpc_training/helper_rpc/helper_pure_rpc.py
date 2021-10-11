from torch.distributed.rpc import RRef, rpc_sync, rpc_async, remote

# https://github.com/pytorch/examples/blob/01539f9eada34aef67ae7b3d674f30634c35f468/distributed/rpc/parameter_server/rpc_parameter_server.py
# --------- Helper Methods --------------------

# On the local node, call a method with first arg as the value held by the
# RRef. Other args are passed in as arguments to the function called.
# Useful for calling instance methods.

def _call_method(method, rref, *args, **kwargs):
    r"""
    rref: remote reference
    a helper function to call a method on the given RRef
    """
    return method(rref.local_value(), *args, **kwargs)

# Given an RRef, return the result of calling the passed in method on the value
# held by the RRef. This call is done on the remote node that owns
# the RRef. args and kwargs are passed into the method.
# Example: If the value held by the RRef is of type Foo, then
# remote_method(Foo.bar, rref, arg1, arg2) is equivalent to calling
# <foo_instance>.bar(arg1, arg2) on the remote node and getting the result
# back.

def _remote_method(method, rref, *args, **kwargs):
    r"""
    a helper function to run method on the owner of rref and fetch back the
    result using RPC
    """
    args = [method, rref] + list(args)
    rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)