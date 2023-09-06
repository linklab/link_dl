import torch

a = torch.ones((2, 3))
print(a.dtype)

b = torch.ones((2, 3), dtype=torch.int16)
print(b)

c = torch.rand((2, 3), dtype=torch.float64) * 20.
print(c)

d = b.to(torch.int32)
print(d)

double_d = torch.ones(10, 2, dtype=torch.double)
short_e = torch.tensor([[1, 2]], dtype=torch.short)

double_d = torch.zeros(10, 2).double()
short_e = torch.ones(10, 2).short()

double_d = torch.zeros(10, 2).to(torch.double)
short_e = torch.ones(10, 2).to(dtype=torch.short)

double_d = torch.zeros(10, 2).type(torch.double)
short_e = torch.ones(10, 2). type(dtype=torch.short)

print(double_d.dtype)
print(short_e.dtype)

double_f = torch.rand(5, dtype=torch.double)
short_g = double_f.to(torch.short)
print((double_f * short_g).dtype)
