class Llama::TransformerFile < BinData
  custom config : TransformerConfig = TransformerConfig.new
  array token_embedding_table : Float32, length: ->{ config.vocab_size * config.dim }
  array rms_att_weight : Float32, length: ->{ config.n_layers * config.dim }
  array wq : Float32, length: ->{ config.n_layers * config.dim * (config.n_heads * (config.dim / config.n_heads)) }
  array wk : Float32, length: ->{ config.n_layers * config.dim * (config.n_kv_heads * (config.dim / config.n_heads)) }
  array wv : Float32, length: ->{ config.n_layers * config.dim * (config.n_kv_heads * (config.dim / config.n_heads)) }
  array wo : Float32, length: ->{ config.n_layers * config.dim * (config.n_heads * (config.dim / config.n_heads)) }
  array rms_ffn_weight : Float32, length: ->{ config.n_layers * config.dim }
  array w1 : Float32, length: ->{ config.n_layers * config.dim * config.hidden_dim }
  array w2 : Float32, length: ->{ config.n_layers * config.dim * config.hidden_dim }
  array w3 : Float32, length: ->{ config.n_layers * config.dim * config.hidden_dim }
  array rms_final_weight : Float32, length: ->{ config.dim + 2 * (config.seq_len * (config.dim / config.n_heads) / 2) }

  remaining_bytes :wcls
end
