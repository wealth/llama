struct Llama::ArrayView(T)
  property array : Array(T)
  property offset : Int32

  def initialize(@array : Array(T), @offset : Int32)
  end

  def [](index : Int32)
    @array[@offset + index]
  end

  def []=(index : Int32, value : T)
    @array[@offset + index] = value
  end
end
