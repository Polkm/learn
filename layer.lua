-- Module for linear transformations using a weight tensor
function learn.layer.linear(p)
  p.n_input = p.n_input or 1
  p.n_output = p.n_output or 1

  p.weight_init = p.weight_init or function()
    return learn.gaussian(0.0, 1.0)
  end

  p.weights = p.weights or learn.tensor({size = {p.n_output, p.n_input}}).map(p.weight_init)
  p.gradients = p.gradients or learn.tensor({size = {p.n_output, p.n_input}})

  function p.forward(input)
    -- print(table.concat(p.weights.data, ", "))
    p.output = p.weights.dot(input)
    -- print(table.concat(p.output.data, ", "))
    return p.output
  end

  function p.backward(input, gradients)
    p.delta = gradients.copy().mul(input)
    return input, p.weights.transpose().dot(p.delta)
  end

  function p.update(input, learning_rate)
    p.weights.sub(p.delta.dot(input.transpose()).scale(learning_rate))
    return p.output
  end

  return p
end
