import torch, time
assert torch.cuda.is_available(), "CUDA not detected by PyTorch"
device = torch.device("cuda:0")
print("Using", torch.cuda.get_device_name(0))

# Matrix size controls GPU memory usage
# 8192 -> ~512MB, 16384 -> ~2GB, 32768 -> ~8GB
SIZE = 16384

# Preallocate tensors on GPU
A = torch.randn(SIZE, SIZE, device=device)
B = torch.randn(SIZE, SIZE, device=device)

while True:
    # Perform multiple matmuls to keep GPU busy
    for _ in range(5):
        C = torch.matmul(A, B)
        A, B = B, C
    torch.cuda.synchronize()
    time.sleep(1)  # Lower = more load
