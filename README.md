Learn's api is something like a combination between [Torch](http://torch.ch/) and [Scikit Learn](http://scikit-learn.org/stable/). The purpose of Learn is to provide a flexible and portable neural network implementation that only depends on Lua. Learn is *not* multithreaded and does *not* use hardware acceleration, if you are looking for a high performance library I would suggest looking at [Torch](http://torch.ch/) instead.

### Start using Learn
Learn can be installed as a submodule for your git project by using the command:
```
git submodule add https://github.com/Polkm/learn.git learn
```

After you have Learn installed you can use it in an existing Lua project.
```lua
require("learn/learn")
```

### Set up your training data
```lua
-- XOR training data
local train_features = {{0, 0}, {0, 1}, {1, 0}, {1, 1}}
local train_labels = {{0}, {1}, {1}, {0}}
```

### Set up your model
```lua
local n_input = train_features[1].size[1]
local n_output = train_labels[1].size[1]

local model = learn.model.nnet({modules = {
  learn.layer.linear({n_input = n_input, n_output = n_input * 3}),
  learn.transfer.sigmoid({}),
  learn.layer.linear({n_input = n_input * 3, n_output = n_output}),
  learn.transfer.sigmoid({}),
}})
```

### Train your model on your data
```lua
model.fit(train_features, train_labels, 1000)
```

### Make predictions using your newly trained model
```lua
for _, prediction in pairs(model.predict(train_features)) do
  print(prediction[1])
end
```
