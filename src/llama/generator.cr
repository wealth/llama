class Llama::Generator
  def initialize(@transformer : Transformer, @tokenizer : Tokenizer, @sampler : Sampler)
  end

  def generate(prompt : String, steps : Int32, &block : String ->)
    raise "steps must be power of 2" if Math.sqrt(steps) % 2 != 0
    prompt = prompt.empty? ? "" : prompt

    # Encode the prompt into tokens sequence
    prompt_tokens = @tokenizer.encode(prompt, true, false)
    raise "something is wrong, expected at least 1 prompt token" if prompt_tokens.empty?

    start = Time.monotonic # Used to time our code
    pos = 0
    token = prompt_tokens[0]

    # output = ""

    while pos < steps
      # Forward the transformer to get logits for the next token
      # time = Benchmark.measure do
      @transformer.forward(token, pos)
      # end
      # puts "\ntotal forward time: #{time}"
      # puts "next logits: #{@transformer.state.logits[0..5]}"
      # puts "sample for this logits: #{@sampler.sample(@transformer.state.logits)}"

      # Advance the state machine
      next_token = (pos < prompt_tokens.size - 1) ? prompt_tokens[pos + 1] : @sampler.sample(@transformer.state.logits)

      break if next_token == 1 # Data-dependent terminating condition

      # Print the token as string, decode it with the Tokenizer object
      piece = @tokenizer.decode(token, next_token)
      # output += piece.unicode_normalize
      yield piece if block && (pos >= prompt_tokens.size - 1)

      token = next_token
      pos += 1
      # Process.exit(0)
    end
    # puts output
    # puts # New line for the end of output

    # Report achieved tokens per second
    end_time = Time.monotonic
    elapsed_seconds = (end_time - start).total_seconds
    tok_per_second = (pos - 1) / elapsed_seconds
    puts "\nachieved tok/s: #{tok_per_second}"
    # output
  end
end
