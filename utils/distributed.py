import os
import torch
import torch.distributed as dist
from typing import List, Union


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            args = ["[Main]"] + list(args)
            builtin_print(*args, **kwargs)
        else:
            args = [f'[Slave#{get_rank()}] '] + list(args)
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args):
    if args.distributed:
        rank = args.local_rank + args.node_rank * args.proc_per_node

        torch.cuda.set_device(args.local_rank)
        args.dist_backend = 'nccl'
        socket_name = os.environ.get("NCCL_SOCKET_IFNAME")

        if args.node == 1:
            args.dist_url = 'tcp://127.0.0.1:23456'
        else:
            if args.dist_url == 'env://':
                args.dist_url = f'tcp://{os.environ.get("MASTER_ADDR")}:{os.environ.get("MASTER_PORT")}'

        print(f'| distributed init (rank {rank}): {args.dist_url}, NCCL_SOCKET_IFNAME: {socket_name}', flush=True)
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                             world_size=args.world_size, rank=rank)
        torch.distributed.barrier()
        # setup_for_distributed(args.rank == 0)

    args.rank = get_rank()
    args.world_size = get_world_size()
    args.device = torch.device(f'cuda:{args.local_rank}')


def gather_object(obj, rank, world, dst=0) -> List[object]:
    dist.barrier()

    received = [None] * world if rank == dst else None
    dist.gather_object(obj, received, dst=dst)
    return received if rank == dst else None
    # received = [None] * world
    # dist.all_gather_object(received, obj)
    return received if rank == dst else None


def gather_list(obj: list, rank, world, dst=0) -> List[object]:
    received = gather_object(obj, rank, world, dst)
    return [item for sublist in received for item in sublist] if rank == dst else None


def gather_tensor(obj, rank, world, dst=0) -> torch.Tensor:
    received = gather_object(obj, rank, world, dst)
    return torch.cat(received, dim=0) if rank == dst else None


def gather_dict(obj: dict, rank, world, dst=0):
    received = gather_object(obj, rank, world, dst)  # list of dictionary
    if rank == dst:
        for d in received[1:]:
            received[0].update(d)
        return received[0]
    else:
        return None
