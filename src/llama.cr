require "bindata"
require "crest"
require "./llama/**"

module Llama
  VERSION = "0.1.0"

  def self.build_simple_generator(
    model_filename : String = "stories15M.bin",
    tokenizer_path : String = "models/tokenizer.bin",
    temperature : Float32 = 1.0_f32,
    topp : Float32 = 0.9_f32,
    rng_seed : UInt64 = 0_u64
  )
    # Example of setting rng_seed based on current time if it's 0
    rng_seed = Time.utc.to_unix.to_u64 if rng_seed == 0_u64

    # Parameter validation/overrides
    rng_seed = Time.utc.to_unix.to_u64 if rng_seed <= 0_u64
    temperature = 0.0 if temperature < 0.0
    topp = 0.9 if topp < 0.0 || topp > 1.0

    checkpoint_path = "models/#{model_filename}"
    if !File.exists?(checkpoint_path)
      puts "Downloading model from tinyllama..."
      download_model(
        "https://huggingface.co/karpathy/tinyllamas/resolve/main/#{checkpoint_path}",
        checkpoint_path)
      raise ArgumentError.new("checkpoint_path does not exist") if !File.exists?(checkpoint_path)
    end
    model_file = File.open(checkpoint_path)
    model = model_file.read_bytes(Llama::TransformerFile)
    puts model.config.pretty_inspect
    transformer = Llama::Transformer.new(model)
    tokenizer = Llama::Tokenizer.new(vocab_size: transformer.model.config.vocab_size)
    tokenizer.build(tokenizer_path)
    sampler = Llama::Sampler.new(
      vocab_size: transformer.model.config.vocab_size,
      temperature: temperature,
      topp: topp,
      rng_state: rng_seed,
    )
    model_file.close
    return Llama::Generator.new(transformer, tokenizer, sampler)
  end

  private def self.download_model(url, path)
    Crest.get(url) do |resp|
      File.open(path, "w") do |file|
        IO.copy(resp.body_io, file)
      end
    end
  end
end
