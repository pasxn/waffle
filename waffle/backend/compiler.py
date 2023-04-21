

class shapetracker:

  shapes = []

  @staticmethod
  def parse(x_shape, y_shape, op_type):
    if x_shape is None: x_dim = 0
    else: x_dim = len(x_shape)
    if y_shape is None: y_dim = 0
    else: y_dim = len(y_shape)
    shape = {
      'x_shape' : x_shape,
      'y_shape' : y_shape,
      'x_dim'   : x_dim,
      'y_dim'   : y_dim,
      'op_type' : (op_type.name).lower()
    }
    
    return shape
  
  @staticmethod
  def fill(shape):
    shapetracker.shapes.append(shape)

  @staticmethod
  def clear():
    shapetracker.shapes.clear()

  @staticmethod
  def pop():
    return shapetracker.shapes.pop()
