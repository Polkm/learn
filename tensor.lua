-- Helper class for 2D matrix math and such
function learn.tensor(p)
  p.data = p.data or {}
  p.size = p.size or {#p.data, 1}

  function p.set(v, x, y)
    p.data[x + (y - 1) * p.size[1]] = v
  end

  function p.get(x, y)
    return p.data[x + (y - 1) * p.size[1]]
  end

  function p.copy()
    return learn.tensor({size = {p.size[1], p.size[2]}}).map(function(v, x, y) return p.get(x, y) end)
  end

  function p.each(func)
    for x = 1, p.size[1] do
      for y = 1, p.size[2] do
        func(p.get(x, y), x, y)
      end
    end
    return p
  end

  function p.map(func)
    return p.each(function(v, x, y) p.set(func(p.get(x, y), x, y), x, y) end)
  end

  function p.add(b)
    return p.map(function(v, x, y) return v + b.get(x, y) end)
  end
  function p.sub(b)
    return p.map(function(v, x, y) return v - b.get(x, y) end)
  end
  function p.div(b)
    return p.map(function(v, x, y) return v / b.get(x, y) end)
  end
  function p.mul(b)
    return p.map(function(v, x, y) return v * b.get(x, y) end)
  end
  function p.scale(s)
    return p.map(function(v, x, y) return v * s end)
  end

  function p.pow(e)
    return p.map(function(v, x, y) return math.pow(v, e) end)
  end

  function p.sum(result)
    result = result or learn.tensor({size = {1, 1}, data = {0}})
    p.each(function(v) result.data[1] = result.data[1] + v end)
    return result
  end

  function p.dot(b, result)
    assert(p.size[2] == b.size[1], "Invalid dot product tensor size " .. p.size[2] .. " " .. b.size[1])

    if result then
      result.size[1], result.size[2] = p.size[1], b.size[2]
    else
      result = learn.tensor({size = {p.size[1], b.size[2]}})
    end

    result.map(function(v, x, y)
      local sum = 0
      for c = 1, p.size[2] do
        sum = sum + p.get(x, c) * b.get(c, y)
      end
      return sum
    end)

    return result
  end

  function p.transpose()
    local q = p.copy()
    q.size = {p.size[2], p.size[1]}
    p.each(function(v, x, y)
      q.set(p.get(x, y), y, x)
    end)
    return q
  end

  function p.string()
    local str = ""
    for x = 1, p.size[1] do
      for y = 1, p.size[2] do
        str = str .. (p.get(x, y) or "nil") .. " "
      end
      str = str .. "\n"
    end
    return str
  end

  return p
end
