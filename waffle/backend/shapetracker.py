class shapetracker:

  shapes = []

  @staticmethod
  def parse(x_shape, y_shape, op_type):
    shape = {
      'x_shape' : x_shape,
      'y_shape' : y_shape,
      'x_dim'   : len(x_shape),
      'y_dim'   : len(y_shape),
      'op_type' : (op_type.name).lower()
    }
    shapetracker.shapes.append(shape)
