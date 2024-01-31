struct Llama::Transformer
  property model : TransformerFile
  property state : RunState

  def initialize(@model : TransformerFile)
    @state = RunState.new(@model.config)
  end

  def forward(token : Int32, pos : Int32) : Array(Float32)
    # puts "starting logits: #{@state.logits[0..5]}"
    dim = @model.config.dim
    hidden_dim = @model.config.hidden_dim
    kv_dim = ((dim * @model.config.n_kv_heads) / @model.config.n_heads).to_i
    kv_mul = (@model.config.n_kv_heads / @model.config.n_heads).to_i
    head_size = (dim / @model.config.n_heads).to_i

    # puts "token: #{token}, pos: #{pos}"
    @state.x.replace(@model.token_embedding_table[token * dim, dim])

    z = Benchmark.measure do
      @model.config.n_layers.times do |layer_idx|
        # puts "xb before rmsnorm: #{@state.xb[0..5]}"
        Llama.rmsnorm(@state.xb, @state.x, @model.rms_att_weight, layer_idx * dim, dim)
        # puts "xb after rmsnorm: #{@state.xb[0..5]}"

        layer_offset = layer_idx * @model.config.seq_len * kv_dim
        @state.k = ArrayView.new(@state.key_cache, layer_offset + pos * kv_dim)
        @state.v = ArrayView.new(@state.value_cache, layer_offset + pos * kv_dim)

        # qkv matmuls for this position
        Llama.matmul(@state.q, @state.xb, ArrayView.new(@model.wq, layer_idx * dim * dim), dim, dim)
        Llama.matmul(@state.k, @state.xb, ArrayView.new(@model.wk, layer_idx * dim * kv_dim), dim, kv_dim)
        Llama.matmul(@state.v, @state.xb, ArrayView.new(@model.wv, layer_idx * dim * kv_dim), dim, kv_dim)

        # RoPE relative positional encoding: complex-valued rotate q and k in each head
        dim.times.step(2).each do |i|
          head_dim = i % head_size
          freq = 1.0_f32 / 10000.0_f32 ** (head_dim / head_size.to_f32)
          val = pos * freq
          fcr = Math.cos(val)
          fci = Math.sin(val)
          rotn = i < kv_dim ? 2 : 1
          rotn.times do |v|
            vec = v == 0 ? @state.q : @state.k
            v0 = vec[i]
            v1 = vec[i + 1]
            vec[i] = v0 * fcr - v1 * fci
            vec[i + 1] = v0 * fci + v1 * fcr
          end
        end

        # Multihead Attention
        mh = Benchmark.measure do
          @model.config.n_heads.times do |head_idx|
            spawn do
              q_offset = head_idx * head_size
              att_offset = head_idx * @model.config.seq_len

              (pos + 1).times do |t|
                k_offset = (layer_offset + t * kv_dim + (head_idx / kv_mul) * head_size).to_i
                score = 0.0_f32
                head_size.times do |i|
                  score += @state.q[q_offset + i] * @state.key_cache[k_offset + i]
                end
                score /= Math.sqrt(head_size)
                @state.att[att_offset + t] = score.to_f32
              end

              # puts "att before softmax: #{@state.att[0..5]}"

              Llama.softmax(@state.att, att_offset, pos + 1)

              # puts "att after softmax: #{@state.att[0..5]}"

              xb_offset = head_idx * head_size
              head_size.times { |i| @state.xb[xb_offset + i] = 0 }
              # puts "xb after memset: #{@state.xb[0..5]}"
              (pos + 1).times do |t|
                v_offset = (layer_offset + t * kv_dim + (head_idx / kv_mul) * head_size).to_i
                a = @state.att[att_offset + t]
                head_size.times do |i|
                  # puts "#{a}, #{@state.value_cache[v_offset + i]}"
                  @state.xb[xb_offset + i] += a * @state.value_cache[v_offset + i]
                end
              end
            end
          end
        end
        Fiber.yield
        puts "multihead time: #{mh}"

        Llama.matmul(@state.xb2, @state.xb, ArrayView.new(@model.wo, layer_idx * dim * dim), dim, dim)

        # puts "xb after matmul xb2: #{@state.xb[0..5]}"
        # puts "xb2 after matmul xb: #{@state.xb2[0..5]}"

        dim.times do |i|
          @state.x[i] += @state.xb2[i]
        end

        # puts "x: #{@state.x[0..5]}"

        Llama.rmsnorm(@state.xb, @state.x, @model.rms_ffn_weight, layer_idx * dim, dim)
        Llama.matmul(@state.hb, @state.xb, ArrayView.new(@model.w1, layer_idx * dim * hidden_dim), dim, hidden_dim)
        Llama.matmul(@state.hb2, @state.xb, ArrayView.new(@model.w3, layer_idx * dim * hidden_dim), dim, hidden_dim)

        hidden_dim.times do |i|
          val = @state.hb[i]
          val *= (1.0_f32 / (1.0_f32 + Math.exp(-val))).to_f32
          val *= @state.hb2[i]
          @state.hb[i] = val
        end

        # puts "xb before matmul: #{@state.xb[0..5]}"
        # puts "hb before matmul: #{@state.hb[0..5]}"
        Llama.matmul(@state.xb, @state.hb, ArrayView.new(@model.w2, layer_idx * dim * hidden_dim), hidden_dim, dim)

        dim.times do |i|
          @state.x[i] += @state.xb[i]
        end

        # puts "xb end of cycle: #{@state.xb[0..5]}"
        # Process.exit(0)
      end
    end
    puts "layers time: #{z}"

    # puts "xb after forward: #{@state.xb[0..5]}"

    z = Benchmark.measure do
      Llama.rmsnorm(@state.x, @state.x, @model.rms_final_weight, 0, dim)
    end
    puts "rmsnorm time: #{z}"
    z = Benchmark.measure do
      Llama.matmul(@state.logits, @state.x, ArrayView.new(@model.token_embedding_table, 0), dim, @model.config.vocab_size)
    end
    puts "logits time: #{z}"

    puts "ending logits: #{@state.logits[0..5]}"
    @state.logits
  end
end
