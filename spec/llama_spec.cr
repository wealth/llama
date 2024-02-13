require "./spec_helper"
require "random"

# require "num"
#
struct StaticArray(T, N)
  def self.new_fast(& : Int32 -> T)
    array = uninitialized self
    buf = array.to_unsafe
    {% for i in 0...N %}
    buf[{{i.id}}] = yield {{i.id}}
    {% end %}
    array
  end
end

describe Llama do
  # TODO: Write tests

  it "works" do
    generator = Llama.build_simple_generator(models_dir: "/tmp/spider-models/", model_filename: "stories15M.bin")
    prompt = "Once upon a time"
    generator.generate(prompt, 256) do |token|
      print token
    end
  end

  # it "matmul on par with c" do
  #   # x = Array(Float32).new(288) { Random.rand.to_f32 }
  #   # w = Array(Float32).new(32000 * 288) { Random.rand.to_f32 }
  #   x = Array(Float32).new(288, 1_f32)
  #   w = Array(Float32).new(9216000, 2_f32)
  #   xout = Array(Float32).new(32000, 0)

  #   xs = Slice(Float32).new(288, 1_f32)
  #   ws = Slice(Float32).new(9216000, 2_f32)
  #   xouts = Slice(Float32).new(32000, 0)

  #   # puts "Allocating..."
  #   xst = StaticArray(Float32, 288).new_fast { Random.rand.to_f32 }
  #   # puts "xst size: #{xst.size}"
  #   # wst = StaticArray(Float32, 9216000).new_fast { Random.rand.to_f32 }
  #   wst = Array(Float32).new(9216000) { Random.rand.to_f32 }
  #   # puts "wst size: #{wst.size}"
  #   xoutst = uninitialized StaticArray(Float32, 32000)
  #   # puts "xoutst size: #{xoutst.size}"

  #   # puts "x slice: #{x[0..10]}, x size #{x.size}"
  #   # puts "w slice: #{w[0..10]}, w size #{w.size}"
  #   # xt = Tensor(Float32, CPU(Float32)).new([288, 1])
  #   # wt = Tensor(Float32, CPU(Float32)).new([32000, 288])
  #   # xoutt = Tensor(Float32, CPU(Float32)).new([32000, 1])
  #   puts "Allocated. Benchmarking matmul..."
  #   Benchmark.ips(warmup: 4, calculation: 10) do |z|
  #     z.report("matmul array") { Llama.matmul(xout, x, w, 288, 32000) }
  #     z.report("matmul slices") { Llama.matmul(xouts, xs, ws, 288, 32000) }
  #     z.report("matmul partial static") { Llama.matmul(xoutst, xst, wst) }
  #     # z.report("matmul static") { Llama.matmul(xoutst, xst, wst) }
  #     # z.report("matmul tensor") { wt.matmul(xt, xoutt) }
  #   end
  #   puts xoutst.sum
  # end
end
