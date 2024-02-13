struct Llama::RunState
  # property x : Array(Float32)
  property x : StaticArray(Float32, 288)

  property xb : Array(Float32)
  property xb2 : Array(Float32)
  property hb : Array(Float32)
  property hb2 : Array(Float32)
  property q : Array(Float32)
  property k : ArrayView(Float32)
  property v : ArrayView(Float32)
  property att : Array(Float32)

  # property logits : Array(Float32)
  property logits : StaticArray(Float32, 32000)

  property key_cache : Array(Float32)
  property value_cache : Array(Float32)

  def initialize(config : TransformerConfig)
    kv_dim = ((config.dim * config.n_kv_heads) / config.n_heads).to_i

    # @x = Array(Float32).new(config.dim, 0.0_f32)
    @x = StaticArray(Float32, 288).new(0.0_f32)

    @xb = Array(Float32).new(config.dim, 0.0_f32)
    @xb2 = Array(Float32).new(config.dim, 0.0_f32)
    @hb = Array(Float32).new(config.hidden_dim, 0.0_f32)
    @hb2 = Array(Float32).new(config.hidden_dim, 0.0_f32)
    @q = Array(Float32).new(config.dim, 0.0_f32)
    @key_cache = Array(Float32).new(config.n_layers * config.seq_len * kv_dim, 0.0_f32)
    @value_cache = Array(Float32).new(config.n_layers * config.seq_len * kv_dim, 0.0_f32)
    @k = ArrayView.new(@key_cache, 0)
    @v = ArrayView.new(@value_cache, 0)
    @att = Array(Float32).new(config.n_heads * config.seq_len, 0.0_f32)

    # @logits = Array(Float32).new(config.vocab_size, 0.0_f32)
    @logits = uninitialized StaticArray(Float32, 32000)
  end
end
