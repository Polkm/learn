function learn.criterion.mse(p)
  function p.forward(predictions, target)
    p.output_tensor = predictions.copy().sub(target).pow(2.0).sum().scale(1.0 / predictions.size[1])
    p.output = p.output_tensor.data[1]
    return p.output
  end

  function p.backward(predictions, target)
    p.gradInput = predictions.copy().sub(target).scale(2.0 / predictions.size[1])
    return p.gradInput
  end

  return p
end
