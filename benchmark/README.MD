# LSH Benchmark

Build image

```
docker build -f lsh_benchmark.dockerfile -t lsh_benchmark .
```

Load with data

```
docker run --rm -v ${PWD}/../data:/data -v ${PWD}/../src/:/home --entrypoint /bin/bash -it lsh_benchmark

# Within Docker container
python3 lsh.py load /data/music_data.npy /data/store.db
```

Run benchmark

```
docker run --rm -v ${PWD}/../data:/data -v ${PWD}/../src/:/home lsh_benchmark
```

Results are now in `benchmark.output.txt` as a CSV file. 
First column is index according to input file of matrices and second column is the result.
1 for input matrix being a song, 0 otherwise.
