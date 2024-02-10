require "./spec_helper"

describe Llama do
  # TODO: Write tests

  it "works" do
    generator = Llama.build_simple_generator(models_dir: "/tmp/spider-models/", model_filename: "stories15M.bin")
    prompt = "Once upon a time"
    generator.generate(prompt, 256) do |token|
      print token
    end
  end
end
