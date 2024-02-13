require "benchmark"

module Llama
  def self.rmsnorm(xout : Array(Float32), x : Array(Float32), weight : Array(Float32), weight_offset : Int32, size : Int32)
    # Calculate sum of squares
    # time = Benchmark.measure do
    ss = 0.0_f32
    size.times do |j|
      ss += x[j] * x[j]
    end
    ss /= size.to_f32
    ss += 1e-5_f32
    ss = 1.0_f32 / Math.sqrt(ss)

    # Normalize and scale
    size.times do |j|
      # puts "xout.size = #{xout.size}, x.size = #{x.size}, weight.size = #{weight.size}"
      xout[j] = weight[weight_offset + j] * (ss * x[j])
    end
    # end
    # puts "rmsnorm time: #{time}"
  end

  def self.rmsnorm(xout : StaticArray(Float32, 288), x : StaticArray(Float32, 288), weight : Array(Float32), weight_offset : Int32, size : Int32)
    # Calculate sum of squares
    # time = Benchmark.measure do
    ss = 0.0_f32
    size.times do |j|
      ss += x[j] * x[j]
    end
    ss /= size.to_f32
    ss += 1e-5_f32
    ss = 1.0_f32 / Math.sqrt(ss)

    # Normalize and scale
    size.times do |j|
      # puts "xout.size = #{xout.size}, x.size = #{x.size}, weight.size = #{weight.size}"
      xout[j] = weight[weight_offset + j] * (ss * x[j])
    end
    # end
    # puts "rmsnorm time: #{time}"
  end

  def self.rmsnorm(xout : Array(Float32), x : StaticArray(Float32, 288), weight : Array(Float32), weight_offset : Int32, size : Int32)
    # Calculate sum of squares
    # time = Benchmark.measure do
    ss = 0.0_f32
    size.times do |j|
      ss += x[j] * x[j]
    end
    ss /= size.to_f32
    ss += 1e-5_f32
    ss = 1.0_f32 / Math.sqrt(ss)

    # Normalize and scale
    size.times do |j|
      # puts "xout.size = #{xout.size}, x.size = #{x.size}, weight.size = #{weight.size}"
      xout[j] = weight[weight_offset + j] * (ss * x[j])
    end
    # end
    # puts "rmsnorm time: #{time}"
  end

  def self.softmax(x : Array(Float32), x_offset : Int32 = 0, x_slice_size : Int32 = x.size)
    # time = Benchmark.measure do
    # Find max value (for numerical stability)
    max_val = x[x_offset + 0]
    (1...x_slice_size).each do |i|
      max_val = x[x_offset + i] if x[x_offset + i] > max_val
    end

    # Exp and sum
    sum = 0.0_f32
    x_slice_size.times do |i|
      x[x_offset + i] = Math.exp(x[x_offset + i] - max_val)
      sum += x[x_offset + i]
    end

    # Normalize
    x_slice_size.times do |i|
      x[x_offset + i] /= sum
    end
    # end
    # puts "softmax time: #{time}"
  end

  def self.softmax(x : StaticArray(Float32, 32000), x_offset : Int32 = 0, x_slice_size : Int32 = x.size)
    # time = Benchmark.measure do
    # Find max value (for numerical stability)
    max_val = x[x_offset + 0]
    (1...x_slice_size).each do |i|
      max_val = x[x_offset + i] if x[x_offset + i] > max_val
    end

    # Exp and sum
    sum = 0.0_f32
    x_slice_size.times do |i|
      x[x_offset + i] = Math.exp(x[x_offset + i] - max_val)
      sum += x[x_offset + i]
    end

    # Normalize
    x_slice_size.times do |i|
      x[x_offset + i] /= sum
    end
    # end
    # puts "softmax time: #{time}"
  end

  def self.matmul(xout : Array(Float32), x : Array(Float32), w : ArrayView(Float32), n : Int32, d : Int32)
    # W (d,n) @ x (n,) -> xout (d,)
    # time = Benchmark.measure do
    d.times do |i|
      # spawn do
      val = 0.0_f32
      idx = i * n
      n.times do |j|
        val += w[idx + j] * x[j]
      end
      xout[i] = val
      # end
    end
    # Fiber.yield
    # end
    # puts "matmul1 of #{d}x#{n} time: #{time}"
  end

  def self.matmul(xout : ArrayView(Float32), x : Array(Float32), w : ArrayView(Float32), n : Int32, d : Int32)
    # W (d,n) @ x (n,) -> xout (d,)
    # time = Benchmark.measure do
    d.times do |i|
      # spawn do
      val = 0.0_f32
      idx = i * n
      n.times do |j|
        val += w[idx + j] * x[j]
      end
      xout[i] = val
      # end
    end
    # Fiber.yield
    # end
    # puts "matmul2 #{d}x#{n} time: #{time}"
  end

  def self.matmul(xout : Array(Float32), x : Array(Float32), w : Array(Float32), n : Int32, d : Int32)
    # W (d,n) @ x (n,) -> xout (d,)
    # time = Benchmark.measure do
    d.times do |i|
      val = 0.0_f32
      idx = i * n
      n.times do |j|
        val += w[idx + j] * x[j]
      end
      xout[i] = val
    end
    # Fiber.yield
    # end
    # puts "matmul3 of #{d}x#{n} time: #{time}"
  end

  def self.matmul(xout : StaticArray(Float32, 32000), x : StaticArray(Float32, 288), w : Array(Float32), n : Int32, d : Int32)
    # W (d,n) @ x (n,) -> xout (d,)
    # time = Benchmark.measure do
    xout.size.times do |i|
      val = 0.0_f32
      idx = i * x.size
      x.size.times do |j|
        val += w[idx + j] * x[j]
      end
      xout[i] = val
    end
    # Fiber.yield
    # end
    # puts "matmul3 of #{d}x#{n} time: #{time}"
  end

  def self.matmul(xout : Slice(Float32), x : Slice(Float32), w : Slice(Float32), n : Int32, d : Int32)
    # W (d,n) @ x (n,) -> xout (d,)
    # time = Benchmark.measure do
    xout.size.times do |i|
      val = 0.0_f32
      idx = i * x.size
      x.size.times do |j|
        val += w[idx + j] * x[j]
      end
      xout[i] = val
    end
    # Fiber.yield
    # end
    # puts "matmul3 of #{d}x#{n} time: #{time}"
  end

  def self.str_lookup(str : String, sorted_vocab : Array(TokenIndex)) : Int32
    # Assuming `sorted_vocab` is already sorted by the `str` property
    index = sorted_vocab.index { |tok| tok.str == str }
    index ? sorted_vocab[index].id : -1
  end

  def self.sample_argmax(probabilities : Array(Float32)) : Int32
    max_i = 0
    max_p = probabilities[0]

    probabilities.each_with_index do |prob, i|
      if prob > max_p
        max_i = i
        max_p = prob
      end
    end

    max_i
  end

  def self.sample_argmax(probabilities : StaticArray(Float32, 32000)) : Int32
    max_i = 0
    max_p = probabilities[0]

    probabilities.each_with_index do |prob, i|
      if prob > max_p
        max_i = i
        max_p = prob
      end
    end

    max_i
  end

  def self.sample_mult(probabilities : Array(Float32), coin : Float32) : Int32
    cdf = 0.0_f32

    probabilities.each_with_index do |prob, i|
      cdf += prob
      return i if coin < cdf
    end

    probabilities.size - 1
  end

  def self.sample_mult(probabilities : StaticArray(Float32, 32000), coin : Float32) : Int32
    cdf = 0.0_f32

    probabilities.each_with_index do |prob, i|
      cdf += prob
      return i if coin < cdf
    end

    probabilities.size - 1
  end

  def self.random_u32(state : UInt64) : UInt32
    # state ^= state >> 12
    # state ^= state << 25
    # state ^= state >> 27
    # puts "state = #{state}"
    # state = state * 0x2545F4914F6CDD1Du64
    # UInt32.new(state >> 32)
    Random.rand(UInt32)
  end

  def self.random_f32(state : UInt64) : Float32
    ((random_u32(state) >> 8).to_f32 / 16777216.0).to_f32
  end

  def self.sample_topp(probabilities : Array(Float32), topp : Float32, coin : Float32) : Int32
    probindex = Array(ProbIndex).new
    # Calculate cutoff based on topp
    cutoff = (1.0_f32 - topp) / (probabilities.size - 1).to_f32
    probabilities.each_with_index do |prob, index|
      probindex << ProbIndex.new(prob, index) if prob >= cutoff
    end

    # Sort probindex in descending order of probabilities
    probindex.sort_by!(&.prob)

    # puts "cutoff: #{cutoff}, probabilities: #{probabilities[0..5]}"

    # Truncate the list where cumulative probability exceeds topp
    cumulative_prob = 0.0_f32
    last_idx = probindex.size - 1
    probindex.each_with_index do |pi, i|
      cumulative_prob += pi.prob
      if cumulative_prob > topp
        last_idx = i
        break
      end
    end

    # Sample from the truncated list
    r = coin * cumulative_prob
    cdf = 0.0_f32
    probindex[0..last_idx].each do |pi|
      cdf += pi.prob
      return pi.index if r < cdf
    end

    # puts "something went wrong, returning last index"
    probindex[last_idx].index
  end

  def self.sample_topp(probabilities : StaticArray(Float32, 32000), topp : Float32, coin : Float32) : Int32
    probindex = Array(ProbIndex).new
    # Calculate cutoff based on topp
    cutoff = (1.0_f32 - topp) / (probabilities.size - 1).to_f32
    probabilities.each_with_index do |prob, index|
      probindex << ProbIndex.new(prob, index) if prob >= cutoff
    end

    # Sort probindex in descending order of probabilities
    probindex.sort_by!(&.prob)

    # puts "cutoff: #{cutoff}, probabilities: #{probabilities[0..5]}"

    # Truncate the list where cumulative probability exceeds topp
    cumulative_prob = 0.0_f32
    last_idx = probindex.size - 1
    probindex.each_with_index do |pi, i|
      cumulative_prob += pi.prob
      if cumulative_prob > topp
        last_idx = i
        break
      end
    end

    # Sample from the truncated list
    r = coin * cumulative_prob
    cdf = 0.0_f32
    probindex[0..last_idx].each do |pi|
      cdf += pi.prob
      return pi.index if r < cdf
    end

    # puts "something went wrong, returning last index"
    probindex[last_idx].index
  end
end
