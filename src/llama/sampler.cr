class Llama::Sampler
  property vocab_size : Int32
  property temperature : Float32
  property topp : Float32
  property rng_state : UInt64

  def initialize(@vocab_size : Int32, @temperature : Float32, @topp : Float32, @rng_state : UInt64)
  end

  def sample(logits : Array(Float32)) : Int32
    next_token = 0

    if @temperature == 0.0_f32
      # Greedy argmax sampling
      next_token = Llama.sample_argmax(logits)
    else
      # Apply the temperature to the logits
      @vocab_size.times do |q|
        logits[q] /= @temperature
      end

      # Apply softmax to the logits to get the probabilities for the next token
      Llama.softmax(logits, 0, @vocab_size)

      # puts "logits after softmax: #{logits[0..5]}"

      # Flip a (float) coin
      coin = Llama.random_f32(@rng_state)

      if @topp <= 0 || @topp >= 1
        # Simply sample from the predicted probability distribution
        next_token = Llama.sample_mult(logits, coin)
      else
        # Top-p (nucleus) sampling
        # puts "logits size: #{logits.size}, vocab_size: #{@vocab_size}"
        next_token = Llama.sample_topp(logits, @topp, coin)
      end
    end

    next_token
  end
end
