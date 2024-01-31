require "./spec_helper"

describe Llama do
  # TODO: Write tests

  it "works" do
    generator = Llama.build_simple_generator(models_dir: "/tmp/spider-models/")
    prompt = "Once upon a time in a galaxy far far away"
    generator.generate(prompt, 16384) do |token|
      print token
    end
  end
end
