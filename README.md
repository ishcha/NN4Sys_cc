# NN4Sys_Benchmark

### create fixed input
- Pensieve
```
cd Models/Pensieve
python shuyi_gen_upper.py
cd ../..
```
- Aurora
```
cd Models/Aurora
python shuyi_gen_upper.py
cd ../..
```
- Decima
```
cd Models/Decima
python shuyi_gen_upper.py
cd ../..
```

### create onnx
- Pensieve
```
cd Models/Pensieve
python export.py
cd ../..
```
- Aurora
```
cd Models/Aurora
python export.py
cd ../..
```
- Decima
```
cd Models/Decima
python export.py
cd ../..
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