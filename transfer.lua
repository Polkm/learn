-- Transfer function module, simply applies the function
function learn.transfer.transfer(p)
  function p.forward(input)
    p.output = input.copy().map(p.transfer)
    return p.output
  end

  function p.backward(input, gradients)
    return p.output.copy().map(p.derivative), gradients
  end

  function p.update(input)
    return p.output
  end

  return p
end

-- The sigmoid transfer function in the form of a module
function learn.transfer.sigmoid(p)
  p = learn.transfer.transfer(p)
  function p.transfer(x) return 1 / (1 + math.exp(-x)) end
  function p.derivative(x) return x * (1 - x) end
  return p
end

-- The hyperbolic tangent transfer function in the form of a module
function learn.transfer.tanh(p)
  p = learn.transfer.transfer(p)
  function p.transfer(x)
    local ex, enx = math.exp(x), math.exp(-x)
    return (ex - enx) / (ex + enx)
  end
  function p.derivative(x)
    local e2x = math.exp(2.0 * x)
    return 4 * e2x / math.pow(e2x + 1, 2.0)
  end
  return p
end

-- Applies the rectified linear unit function
function learn.transfer.relu(p)
  p = learn.transfer.transfer(p)
  function p.transfer(x) return math.max(0, x) end
  function p.derivative(x) return x > 0 and 1 or 0 end
  return p
end
