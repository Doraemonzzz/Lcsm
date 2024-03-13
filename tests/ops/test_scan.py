import pytest
import torch
import torch.nn.functional as F

from mnet_pytorch.ops import (
    expand_and_shrink,
    pscan,
    pscan_block,
    pscan_cuda_fn,
    pscan_torch,
)


@pytest.mark.parametrize(
    "b, n, k, d",
    [
        # (6, 512, 32, 128),
        (6, 2048, 32, 128),
        # (6, 256, 32, 128),
    ],
)
@pytest.mark.parametrize(
    "e_dependent, f_dependent, f_learned, s_dependent",
    [
        # (True, True, True, True),     # ok
        # (True, True, True, False),      # ok
        # (True, False, True, True),    # ok
        # (True, False, True, False),    # ok
        # (False, True, True, True),    # ok
        # (False, True, True, False),     # ok
        # (False, False, True, True),   # ok
        # (False, False, True, False),  # ok
        # (True, False, False, True),    # ok
        (False, False, True, False),  # ok
    ],
)
# @pytest.mark.parametrize("mimo", [False])
@pytest.mark.parametrize("mimo", [True])
# @pytest.mark.parametrize('dtype', [torch.float32])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_op(
    b,
    n,
    k,
    d,
    e_dependent,
    f_dependent,
    f_learned,
    s_dependent,
    dtype,
    mimo,
    device="cuda:0",
):
    torch.manual_seed(20)
    i = (
        torch.empty((b, n, d), dtype=dtype, device=device)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    if e_dependent:
        e = (
            torch.empty((b, n, k), dtype=dtype, device=device)
            .normal_(mean=0.0, std=0.5)
            .requires_grad_()
        )
    else:
        if mimo:
            shape = (k, d)
        else:
            shape = (k,)
        e = (
            torch.empty(shape, dtype=dtype, device=device)
            .normal_(mean=0.0, std=0.5)
            .requires_grad_()
        )

    if f_dependent:
        f = F.sigmoid(
            torch.empty((b, n, k, d), dtype=dtype, device=device).normal_(
                mean=0.0, std=0.5
            )
        ).requires_grad_()
    else:
        if f_learned:
            f = F.sigmoid(
                torch.empty((k, d), dtype=dtype, device=device).normal_(
                    mean=0.0, std=0.5
                )
            ).requires_grad_()
        else:
            f = F.sigmoid(
                torch.empty((k, d), dtype=dtype, device=device).normal_(
                    mean=0.0, std=0.5
                )
            )

    if s_dependent:
        s = (
            torch.empty((b, n, k), dtype=dtype, device=device)
            .normal_(mean=0.0, std=0.5)
            .requires_grad_()
        )
    else:
        if mimo:
            shape = (k, d)
        else:
            shape = (k,)
        s = (
            torch.empty(shape, dtype=dtype, device=device)
            .normal_(mean=0.0, std=0.5)
            .requires_grad_()
        )

    dout = torch.randn(b, n, d).to(i.device)

    # reference
    ref_out = expand_and_shrink(i, e, f, s)
    ref_out.backward(dout, retain_graph=True)
    ref_di, i.grad = i.grad.clone(), None
    ref_de, e.grad = e.grad.clone(), None
    if f_learned:
        ref_df, f.grad = f.grad.clone(), None
    ref_ds, s.grad = s.grad.clone(), None

    # pscan
    pscan_out = pscan(i, e, f, s)
    pscan_out.backward(dout, retain_graph=True)
    pscan_di, i.grad = i.grad.clone(), None
    pscan_de, e.grad = e.grad.clone(), None
    if f_learned:
        pscan_df, f.grad = f.grad.clone(), None
    pscan_ds, s.grad = s.grad.clone(), None

    if not mimo:
        # pscan torch
        pscan_torch_out = pscan_torch(i, e, f, s)
        pscan_torch_out.backward(dout, retain_graph=True)
        pscan_torch_di, i.grad = i.grad.clone(), None
        pscan_torch_de, e.grad = e.grad.clone(), None
        if f_learned:
            pscan_torch_df, f.grad = f.grad.clone(), None
        pscan_torch_ds, s.grad = s.grad.clone(), None

        # pscan block
        pscan_block_out = pscan_block(i, e, f, s)
        pscan_block_out.backward(dout, retain_graph=True)
        pscan_block_di, i.grad = i.grad.clone(), None
        pscan_block_de, e.grad = e.grad.clone(), None
        if f_learned:
            pscan_block_df, f.grad = f.grad.clone(), None
        pscan_block_ds, s.grad = s.grad.clone(), None

    # pscan cuda
    pscan_cuda_out = pscan_cuda_fn(i, e, f, s)
    pscan_cuda_out.backward(dout, retain_graph=True)
    pscan_cuda_di, i.grad = i.grad.clone(), None
    pscan_cuda_de, e.grad = e.grad.clone(), None
    if f_learned:
        pscan_cuda_df, f.grad = f.grad.clone(), None
    pscan_cuda_ds, s.grad = s.grad.clone(), None

    print("naive Vs pscan")
    print(f"out: {torch.norm(ref_out.float() - pscan_out.float())}")
    print(f"di: {torch.norm(ref_di.float() - pscan_di.float())}")
    print(f"de: {torch.norm(ref_de.float() - pscan_de.float())}")
    if f_learned:
        print(f"df: {torch.norm(ref_df.float() - pscan_df.float())}")
    print(f"ds: {torch.norm(ref_ds.float() - pscan_ds.float())}")

    if not mimo:
        print("naive Vs pscan torch")
        print(f"out: {torch.norm(ref_out.float() - pscan_torch_out.float())}")
        print(f"di: {torch.norm(ref_di.float() - pscan_torch_di.float())}")
        print(f"de: {torch.norm(ref_de.float() - pscan_torch_de.float())}")
        if f_learned:
            print(f"df: {torch.norm(ref_df.float() - pscan_torch_df.float())}")
        print(f"ds: {torch.norm(ref_ds.float() - pscan_torch_ds.float())}")

        print("naive Vs pscan block")
        print(f"out: {torch.norm(ref_out.float() - pscan_block_out.float())}")
        print(f"di: {torch.norm(ref_di.float() - pscan_block_di.float())}")
        print(f"de: {torch.norm(ref_de.float() - pscan_block_de.float())}")
        if f_learned:
            print(f"df: {torch.norm(ref_df.float() - pscan_block_df.float())}")
        print(f"ds: {torch.norm(ref_ds.float() - pscan_block_ds.float())}")

    print("naive Vs pscan cuda")
    print(f"out: {torch.norm(ref_out.float() - pscan_cuda_out.float())}")
    print(f"di: {torch.norm(ref_di.float() - pscan_cuda_di.float())}")
    print(f"de: {torch.norm(ref_de.float() - pscan_cuda_de.float())}")
    if f_learned:
        print(f"df: {torch.norm(ref_df.float() - pscan_cuda_df.float())}")
    print(f"ds: {torch.norm(ref_ds.float() - pscan_cuda_ds.float())}")

    torch.testing.assert_close(ref_out.float(), pscan_out.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ref_di.float(), pscan_di.float(), atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(ref_de.float(), pscan_de.float(), atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(ref_df.float(), pscan_df.float(), atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(ref_ds.float(), pscan_ds.float(), atol=5e-2, rtol=1e-2)
