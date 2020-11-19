import torch

# 如果设置 .requires_grad 为 True，那么将会追踪所有对于该张量的操作。 当完成计算后通过调用 .backward()，自动计算所有的梯度， 这个张量的所有梯度将会自动积累到 .grad 属性。
# 要阻止张量跟踪历史记录，可以调用.detach()方法将其与计算历史记录分离，并禁止跟踪它将来的计算记录。
# 为了防止跟踪历史记录（和使用内存），可以将代码块包装在with torch.no_grad()：中。 在评估模型时特别有用，因为模型可能具有requires_grad = True的可训练参数，但是我们不需要梯度计算。
# 在自动梯度计算中还有另外一个重要的类Function.

x = torch.ones(2, 2, requires_grad=True)  # 创建一个张量并设置 requires_grad=True 用来追踪他的计算历史，
# 从这里开始x就是一个自变量了

print("x is :")
print(x)
y = x + 2
print("y is :")
print(y)  # 结果y已经被计算出来了，所以，grad_fn已经被自动生成了
print(y.grad_fn)  # 查看具体的信息
print(y.mean())
z = y * y * 3
out = z.mean()
print(out)

out.backward()
print(out)
print(x)
print(x.grad)
print("追踪操作并求导到这里完成")
print("*********************")

xx = torch.Tensor([12.23, 20.95, 32.00])
xx.requires_grad_(True)
y = xx * 2
print(y.data.norm())  # 2范数
while y.data.norm() < 1000:
    y = y * 2
print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)  # v是y的标量函数的梯度
y.backward(v)
print(xx.grad)
print("vector-Jacobian product")
print("********************************")