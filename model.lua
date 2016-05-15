function learn.model.nnet(p)
  p.n_input = p.n_input or 1
  p.n_output = p.n_output or 1

  p.criterion = p.criterion or learn.criterion.mse({})

  function p.forward(input)
    for _, mod in ipairs(p.modules) do
      input = mod.forward(input)
    end
    return input
  end

  function p.backward(input, gradients)
    for i = #p.modules, 1, -1 do
      input, gradients = p.modules[i].backward(input, gradients)
    end
  end

  function p.update(input)
    for i, mod in ipairs(p.modules) do
      input = mod.update(input)
    end
  end

  function p.fit(features, labels, epochs, verbose)
    for e = 1, epochs do
      local error_sum = 0

      for i, input in ipairs(features) do
        input = learn.tensor({data = input})
        local target = learn.tensor({data = labels[i]})

        local output = p.forward(input)

        error_sum = error_sum + p.criterion.forward(output, target)

        p.backward(input, p.criterion.backward(output, target))
        p.update(input)
      end

      if verbose and (e % (epochs / 10)) == 0 then
        print("Error " .. math.floor(error_sum / #features / 0.00001) * 0.00001)
      end
    end
  end

  function p.predict(features)
    local predictions = {}

    for i, feature_vector in ipairs(features) do
      predictions[i] = p.forward(learn.tensor({data = feature_vector})).data
    end

    return predictions
  end

  return p
end
