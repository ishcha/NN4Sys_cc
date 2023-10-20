# NN4Sys_Benchmark


### Train the model
Models are put in /Models dir. We provide training script for pensieve,
decima and aurora. 

### Create fixed input
Install necessary dependencies, then run
```
cd Models
python gen_upper.py
cd ..
```
### Create onnx models
run 
```
cd Models
python export.py
cd ..
```

### Create specifications
run
```
cd Benchmarks
python generate_properties.py
cd ..
```

### Verify with alpha-beta-crown
run
```
cd Verification
python abcrown_run.py
cd ..
```

### Verify with marabou
run
```
cd Verification
python marabou_run.py
cd ..
```

### Draw figures
run
```
cd Verification/figures
python create_json.py
python draw.py
cd ../..
```