struct Llama::Tokenizer
  property vocab : Array(String)
  property vocab_scores : Array(Float32)
  property vocab_size : Int32
  property max_token_length : UInt32
  property sorted_vocab : Array(TokenIndex)
  property byte_pieces : Array(UInt8)

  def initialize(
    @vocab : Array(String) = Array(String).new,
    @vocab_scores : Array(Float32) = Array(Float32).new,
    @sorted_vocab : Array(TokenIndex) = Array(TokenIndex).new,
    @vocab_size : Int32 = 0,
    @max_token_length : UInt32 = 0_u32,
    @byte_pieces : Array(UInt8) = Array(UInt8).new(512, 0_u8)
  )
  end

  def build(tokenizer_path : String)
    File.open(tokenizer_path, mode: "rb") do |file|
      @max_token_length = file.read_bytes(UInt32)
      @vocab_size.times do |i|
        @vocab_scores << file.read_bytes(Float32)
        len = file.read_bytes(Int32)
        vocab_item = Slice(UInt8).new(len)
        res = file.read_fully(vocab_item)
        @vocab << String.new(vocab_item)
      end
      @sorted_vocab = @vocab.map_with_index { |str, id| TokenIndex.new(str, id) }.sort_by(&.str)
    rescue ex
      raise "Failed to load tokenizer from #{tokenizer_path}: #{ex.message}"
    end
  end

  def decode(prev_token : Int32, token : Int32) : String
    piece = @vocab[token]

    # Strip leading whitespace if previous token is BOS (1)
    piece = piece.lstrip if prev_token == 1

    # Parse the token for raw byte patterns like "<0x01>"
    if piece =~ /\A<0[xX]([0-9a-fA-F]{2})>\z/
      byte_val = $1.to_i(16).to_u8
      return byte_val.chr.to_s
    end

    piece
  end

  def encode(text : String, bos : Bool, eos : Bool) : Array(Int32)
    raise "cannot encode NULL text" if text.nil?

    tokens = Array(Int32).new
    tokens << 1 if bos # Prepend BOS token if requested

    # Add dummy prefix if text is not empty (similar logic to C version)
    unless text.empty?
      dummy_prefix = Llama.str_lookup(" ", @sorted_vocab)
      tokens << dummy_prefix unless dummy_prefix == -1
    end

    text.each_char.with_index do |char, index|
      str_buffer = char.to_s
      next_char = text[index + 1]? || '\0'

      # Handle UTF-8 continuation bytes
      if next_char.bytes[0] & 0xC0 == 0x80
        # Append continuation bytes to form the complete UTF-8 character
        while next_char && (next_char.bytes[0] & 0xC0 == 0x80)
          str_buffer += next_char.to_s
          index += 1
          next_char = text[index + 1]?
        end
      end

      # Look up the token
      id = Llama.str_lookup(str_buffer, @sorted_vocab)
      if id != -1
        tokens << id
      else
        # Handle byte fallback encoding
        str_buffer.each_byte do |byte|
          tokens << byte + 3 # Adjusting for control tokens
        end
      end
    end

    # Attempt to merge tokens based on vocab scores (simplified for clarity)
    # The detailed merging logic will depend on the specific scoring and merging strategy

    tokens << 2 if eos # Append EOS token if requested

    tokens
  end
end
