### pytorch 笔记

------

#### ndarray 矩阵：

生成矩阵：

```python
np.array([[1,  2, 3,  4]])						# 初始化
np.random.randn(height, width)					# 生成一个固定宽高的随机矩阵
```

矩阵操作：

```python
np.sum(ndarray)			# 矩阵数值求总和
a[a <= 0] = 0			# 布尔引索（还有切片）
```

#### tensor 张量：

生成张量矩阵：

```python
height, width = 4, 5
type = torch.float
x = torch.zeros(height, width, dtype = type)	# 全 0
x = torch.rand(height, width)					# 随机
x.new_ones(height, width)						# 以 x 为原本制作全 1
torch.randn_like(x, dtype = torch.float)		# 以 x 为原本制作随机
print(x.size(), x.shape)        				# 返回 height, width
```

利用 view 函数重塑张量：

```python
x.view(height, width) 			# 将 x 重塑成 (height, width) 的矩阵
```

ndarray 和 tensor 相互转化：

```python
torch.from_numpy(ndarray)    	# ndarray 转 tensor
tensor.numpy()					# tensor 转 ndarray
```

使用 CUDA (CPU 和 GPU 操作)：

```python
x = torch.rand(height, width)
if torch.cuda.is_available():			# 检查 CUDA 是否可用
    device = torch.device("cuda")
    # 搬入 GPU 的两种方法
    y = torch.ones_like(x, device = device)
    x = x.to(device)
    z = x + y
    print(z)
    print(x.to("cpu", torch.double))    # 搬回 CPU 以供处理 (在 CPU 才能处理)
```