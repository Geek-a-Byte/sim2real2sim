import onnxruntime as ort
import numpy as np
import time

def benchmark_onnx_latency(onnx_path, obs_shape, num_iterations=1000):
    """
    Benchmarks CPU inference latency for the ONNX policy.
    """
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # Warmup
    dummy_input = np.random.randn(1, *obs_shape).astype(np.float32)
    for _ in range(100):
        session.run(None, {input_name: dummy_input})
        
    # Benchmark
    latencies = []
    for _ in range(num_iterations):
        obs = np.random.randn(1, *obs_shape).astype(np.float32)
        
        start_time = time.perf_counter()
        _ = session.run(None, {input_name: obs})
        end_time = time.perf_counter()
        
        latencies.append((end_time - start_time) * 1000) # Convert to ms
        
    avg_latency = np.mean(latencies)
    p99_latency = np.percentile(latencies, 99)
    
    print(f"--- ONNX Latency Benchmark ({onnx_path}) ---")
    print(f"Average Inference Time: {avg_latency:.3f} ms")
    print(f"99th Percentile Time:   {p99_latency:.3f} ms")
    
    target_ms = (1.0 / 30.0) * 1000
    print(f"Control Period (30Hz):  {target_ms:.3f} ms")
    if p99_latency < (target_ms * 0.1): # Target < 10% of control period
        print("Status: PASS (Well under control period)")
    else:
        print("Status: WARNING (Approaching control period limit)")

if __name__ == "__main__":
    benchmark_onnx_latency("deploy/gk_policy.onnx", obs_shape=(6,))