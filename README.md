## Start using Learn
Learn can be installed as a submodule for your git project by useing the command:
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
```

### Set up your model
```lua
local n_input = #train_features[1].data
local n_output = #train_labels[1].data

local model = learn.nnet({criterion = learn.criterion.mse({}), modules = {
  learn.module.linear({n_input = n_input, n_output = n_input * 5.0}),
  learn.module.sigmoid({}),
  learn.module.linear({n_input = n_input * 5.0, n_output = n_input * 10.0}),
  learn.module.sigmoid({}),
  learn.module.linear({n_input = n_input * 10.0, n_output = n_output}),
  learn.module.sigmoid({}),
}})
```

### Train your model on your training data
```lua
model.fit(train_features, train_labels, 1000)
```

### Make predictions using your newly trained model
```lua
local predictions = model.predict(train_features)
for _, p in pairs(predictions) do
  print(p.data[1])
end
```
