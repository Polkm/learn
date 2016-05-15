learn = {}
learn.model = {}
learn.layer = {}
learn.transfer = {}
learn.criterion = {}

-- Returns a random sample of a gaussian distribution
function learn.gaussian(mean, sd)
  return math.sqrt(-2 * math.log(math.random())) * math.cos(2 * math.pi * math.random()) * sd + mean
end

require("learn/tensor")
require("learn/model")
require("learn/layer")
require("learn/transfer")
require("learn/criterion")

function learn.normalize(samples)
  local max = 0
  for i, vector in ipairs(samples) do
    for j, v in ipairs(vector) do
      max = math.max(max, math.abs(v))
    end
  end
  if max  > 0 then
    for i, vector in ipairs(samples) do
      for j, v in ipairs(vector) do
        vector[j] = vector[j] / max
      end
    end
  end
  return max
end

function learn.unormalize(samples, max)
  for i, vector in ipairs(samples) do
    for j, v in ipairs(vector) do
      vector[j] = vector[j] * max
    end
  end
end

-- Runs all unit tests
function learn.test()
  local identity = learn.tensor({data = {1, 0, 0, 1}, size = {2, 2}})
  local test = learn.tensor({data = {1, 2, 3, 4, 5, 6}, size = {2, 3}})
  -- identity.dot(test).print()

  -- print(test.string())
  -- print(test.transpose().string())

  -- XOR training data
  -- local train_features = {{0, 0}, {0, 1}, {1, 0}, {1, 1}}
  -- local train_labels = {{0}, {1}, {1}, {0}}

  local train_features = {{0, 0}, {0, 1}, {1, 0}, {-1, -1}}
  local train_labels = {{0, 0}, {0, 1}, {1, 0}, {3, 3}}
  -- local train_labels = {{0}, {0.1}, {0.3}, {-3.0}}

  local n_input = #train_features[1]
  local n_output = #train_labels[1]

  local model = learn.model.nnet({modules = {
    learn.layer.linear({n_input = n_input, n_output = n_input * 3}),
    learn.transfer.tanh({}),
    learn.layer.linear({n_input = n_input * 3, n_output =  n_output}),
    learn.transfer.tanh({}),
    -- learn.layer.linear({n_input = n_output, n_output = n_output}),
    -- learn.transfer.sigmoid({}),
  }})
  -- 
  -- local epochs = 1000
  -- local learning_rate = 0.5
  -- local error = model.fit(train_features, train_labels, epochs, learning_rate, true)
  --
  -- local predictions = model.predict(train_features)
  --
  -- for _, prediction in pairs(predictions) do
  --   print(table.concat(prediction, ", "))
  -- end
end
