# LSH Benchmark

Build image

```
docker build -f lsh_benchmark.dockerfile -t lsh_benchmark ..
```

Load with data and run benchmark

```
docker run --rm -it --entrypoint /bin/bash lsh_benchmark

# Within Docker container
python3 lsh_benchmark.py
```
