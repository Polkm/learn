-- Module for linear transformations using a weight tensor
function learn.module.linear(p)
  p.n_input = p.n_input or 1
  p.n_output = p.n_output or 1

  p.weight_init = p.weight_init or function()
    return learn.gaussian(0.0, 1.0)
  end

  p.weights = p.weights or learn.tensor({size = {p.n_output, p.n_input}}).map(p.weight_init)
  p.gradients = p.gradients or learn.tensor({size = {p.n_output, p.n_input}})

  function p.forward(input)
    p.output = p.weights.dot(input, p.output)
    return p.output
  end

  function p.backward(pass_back, pass_back_error, pass_output)
    p.delta = pass_back_error.copy().mul(pass_back)

    pass_back_error = p.weights.transpose().dot(p.delta)

    return pass_back, pass_back_error
  end

  function p.update()

    p.weights.add(p.delta.dot(pass_output.transpose()))
  end

  return p
end
