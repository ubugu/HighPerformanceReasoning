package mydomain;


public class CircularBuffer<T> {
	public int begin;
	public int end;
	public int size;
	public T[] buffer;
	
	public CircularBuffer(int size, T[] buffer) {
		this.size = size;
		this.begin = 0;
		this.end = 0;
		this.buffer = buffer;
	}

}
