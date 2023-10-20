# NN4Sys_Benchmark


### Train the model
Models are put in /Models dir. Currently we provide training script for pensieve,
decima and aurora.

### create fixed input
```
cd Models
python gen_upper.py
```

### create vnnlib(for abcrown) and txt(for marabou)
- Pensieve
```
cd Benchmarks
python shuyi_gen.py 1
cd ..
```
- Aurora
```
cd Benchmarks
python aurora_gen.py 1
cd ..
```
- Decima
```
cd Benchmarks
python decima_gen.py 1
cd ..
```

### run abcrown
- Pensieve
- Aurora
- Decima

### run marabou
- Pensieve
- Aurora
- Decima

### parse abcrown result
- Pensieve
- Aurora
- Decima

### parse marabou result
- Pensieve
- Aurora
- Decima

### Draw graph