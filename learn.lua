learn = {}
learn.module = {}
learn.criterion = {}

-- Returns a random sample of a gaussian distribution
function learn.gaussian(mean, sd)
  return math.sqrt(-2 * math.log(math.random())) * math.cos(2 * math.pi * math.random()) * sd + mean
end

require("learn/tensor")
require("learn/layers")

function learn.nnet(p)
  p.n_input = p.n_input or 1
  p.n_output = p.n_output or 1

  function p.forward(input)
    for _, mod in ipairs(p.modules) do
      input = mod.forward(input)
    end
    return input
  end

  function p.backward(input, gradients)
    for i = #p.modules, 1, -1 do
      p.modules[i].backward(input, gradients)
    end
  end

  function p.update()
    for _, mod in ipairs(p.modules) do
      mod.update()
    end
  end

  function p.fit(features, labels, epochs)
    for e = 1, epochs do
      local error_sum = 0

      for i, input in ipairs(features) do
        local output = labels[i]

        local forward_output = p.forward(input)

        -- feed it to the neural network and the criterion
        p.criterion.forward(forward_output, output)

        -- train over this example in 3 steps
        -- (1) zero the accumulation of the gradients
        -- module:zeroGradParameters()
        -- (2) accumulate gradients
        p.backward(input, p.criterion.backward(forward_output, output))
        -- (3) update parameters with a 0.01 learning rate
        -- module:updateParameters(0.01)

        p.update()




        -- local final_output = p.forward(input)
        --
        -- local pass_back = final_output
        -- local pass_back_error = p.criterion.forward(final_output, labels[i].copy())
        --
        -- error_sum = error_sum + (math.abs(pass_back_error.data[1]) + math.abs(pass_back_error.data[2])) / 2
        --
        -- for i = #p.modules, 1, -1 do
        --   local pass_output = input
        --   if p.modules[i - 1] then
        --     pass_output = p.modules[i - 1].output
        --   end
        --
        --   pass_back, pass_back_error = p.modules[i].backward(pass_back, pass_back_error, pass_output)
        --
        -- end
      end

      -- if (e % (epochs / 10)) == 0 then
      --   print("Error " .. math.floor(error_sum / #features / 0.0000001) * 0.0000001)
      -- end
    end
  end

  function p.predict(features)
    local predictions = {}

    for i, feature_vector in ipairs(features) do
      predictions[i] = p.forward(feature_vector)
    end

    return predictions
  end

  return p
end

-- Runs all unit tests
function learn.test()
  local identity = learn.tensor({data = {1, 0, 0, 1}, size = {2, 2}})
  local test = learn.tensor({data = {1, 2, 3, 4, 5, 6}, size = {2, 3}})
  -- identity.dot(test).print()

  -- print(test.string())
  -- print(test.transpose().string())

  -- XOR training data
  local train_features = {
    learn.tensor({data = {0, 0}}),
    learn.tensor({data = {0, 1}}),
    learn.tensor({data = {1, 0}}),
    learn.tensor({data = {1, 1}}),
  }
  local train_labels = {
    learn.tensor({data = {0}}),
    learn.tensor({data = {1}}),
    learn.tensor({data = {1}}),
    learn.tensor({data = {0}}),
  }

  local n_input = #train_features[1].data
  local n_output = #train_labels[1].data

  local net = learn.nnet({criterion = learn.criterion.mse({}), modules = {
    learn.module.linear({n_input = n_input, n_output = n_input * 5.0}),
    learn.module.sigmoid({}),
    learn.module.linear({n_input = n_input * 5.0, n_output = n_input * 10.0}),
    learn.module.sigmoid({}),
    learn.module.linear({n_input = n_input * 10.0, n_output = n_output}),
    learn.module.sigmoid({}),
  }})

  net.fit(train_features, train_labels, 1000)
  
  local predictions = net.predict(train_features)
  for _, p in pairs(predictions) do
    print(p.data[1])
  end



end
