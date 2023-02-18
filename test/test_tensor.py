from waffle.base import tensor

x = tensor([0, 1, 2, 3, 4, 5, 6, 7, 8,9])
for i in range(x.len):
    print(x.data[i])
