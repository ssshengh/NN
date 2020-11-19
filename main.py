# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 Double Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

import torch

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print('hello')
    print(torch.version)
    X = torch.empty(5, 3)
    print(X)
    y = torch.randn(5, 3)
    print(y)
    z = torch.zeros(5, 3)
    print(torch.add(y, z))

    x = torch.tensor([[8, 7, 4],
                      [5, 6, 1]], dtype=torch.float)
    print(x.size()[:1])

    y = torch.randn(1, 1, 5, 5)
    print(y)
