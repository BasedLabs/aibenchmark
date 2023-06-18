# aibenchmark
Benchmark your model against popular benchmarks

1) Select 4 popular benchmarks
2) Implement dataset loading
3) Implement preprocess, postprocess function execution
4) Implement data pass through the model
5) Implement benchmark output

from aibenchmark import Benchmark

```
def preprocess_func(input_instance):
  #tokenise, rescale ...

  return model_input
```

```
def post_process_func(model_output):
  // Decode, map to labels...

  return correct_output
```

```
benchmark = Benchmark("Benchmark_name", model, preprocess_func)
print(benchmark.output_format)
benchmark.run()
```


