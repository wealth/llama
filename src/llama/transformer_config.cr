class Llama::TransformerConfig < BinData
  int32 :dim
  int32 :hidden_dim
  int32 :n_layers
  int32 :n_heads
  int32 :n_kv_heads
  int32 :vocab_size
  int32 :seq_len
end
