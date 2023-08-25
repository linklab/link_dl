import torch

t1 = torch.randint(low=10, high=20, size=(1, 2))
print(t1)

t2 = torch.rand(size=(1, 3))
print(t2)

t3 = torch.randn(size=(1, 3))
print(t3)

t4 = torch.normal(mean=10.0, std=1.0, size=(3, 2))
print(t4)

t5 = torch.linspace(start=0.0, end=5.0, steps=3)
print(t5)

t6 = torch.arange(5)
print(t6)

print("#" * 30)

torch.manual_seed(1729)
random1 = torch.rand(2, 3)
print(random1)

random2 = torch.rand(2, 3)
print(random2)

print()

torch.manual_seed(1729)
random3 = torch.rand(2, 3)
print(random3)

random4 = torch.rand(2, 3)
print(random4)
