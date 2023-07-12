from waffle import tensor


a = tensor.randn(128, 64, 64, 128)
b = tensor.randn(128, 64, 64, 128)

c = a + b

c_flattened = (a.flatten() + b.flatten()).reshape(128, 64, 64, 128)

for i in range(128):
  for j in range(64):
    for k in range(64):
      for l in range(128):
        assert c.data[i][j][k][l] == c_flattened.data[i][j][k][l], "reshaping does not work!"

