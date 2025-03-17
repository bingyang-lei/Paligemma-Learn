import torch
torch.manual_seed(42)
# torch.flatten()
# # 指定形状的元组
# shape = (2, 2, 2,2,2)

# # 创建一个形状为 `shape` 的随机张量，值在0到1之间
# random_tensor = torch.rand(shape)

# print(random_tensor.shape)

# print(random_tensor.flatten(2).shape)
x = torch.arange(6).view(2, 3)  # 2D张量，形状为 (2, 3)
x_t = x.t()                     # 转置后的张量，形状为 (3, 2)

# 尝试对不连续张量使用 view() 会报错
try:
    y = x_t.view(6)
except RuntimeError as e:
    print(e)  # 输出：view size is not compatible with input tensor's size and stride

# 使用 reshape() 或 contiguous() 解决
y = x_t.reshape(6)  # 形状为 (6,)
print(y)           # 输出：tensor([0, 3, 1, 4, 2, 5])