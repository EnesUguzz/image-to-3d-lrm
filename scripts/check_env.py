"""GPU/torch ortam dogrulama. RTX 5080 sm_120 gercekten calisiyor mu."""
import torch


def main():
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    assert torch.cuda.is_available(), "CUDA yok - cu128 kurulumunu kontrol et"
    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print("gpu:", name, "capability:", cap)
    # sm_120 (Blackwell) gercek matmul: eski wheel burada patlar
    a = torch.randn(2048, 2048, device=dev)
    b = torch.randn(2048, 2048, device=dev)
    c = (a @ b).sum().item()
    print("gpu matmul ok, sum=", c)
    # bf16 destegi
    x = torch.randn(64, 64, device=dev, dtype=torch.bfloat16)
    (x @ x).sum().item()
    print("bf16 ok")


if __name__ == "__main__":
    main()
