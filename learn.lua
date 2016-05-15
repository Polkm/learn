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

-- Runs all unit tests
function learn.test()
  local identity = learn.tensor({data = {1, 0, 0, 1}, size = {2, 2}})
  local test = learn.tensor({data = {1, 2, 3, 4, 5, 6}, size = {2, 3}})
  -- identity.dot(test).print()

  -- print(test.string())
  -- print(test.transpose().string())

  -- XOR training data
  local train_features = {{0, 0}, {0, 1}, {1, 0}, {1, 1}}
  local train_labels = {{0}, {1}, {1}, {0}}

  local n_input = #train_features[1]
  local n_output = #train_labels[1]

  local model = learn.model.nnet({modules = {
    learn.layer.linear({n_input = n_input, n_output = n_input * 3}),
    learn.transfer.sigmoid({}),
    learn.layer.linear({n_input = n_input * 3, n_output = n_output}),
    learn.transfer.sigmoid({}),
  }})

  model.fit(train_features, train_labels, 1000, true)

  for _, prediction in pairs(model.predict(train_features)) do
    print(prediction[1])
  end
end
