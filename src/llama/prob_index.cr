struct Llama::ProbIndex
  property prob : Float32
  property index : Int32

  def initialize(@prob : Float32, @index : Int32)
  end
end
