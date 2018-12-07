### Start the FPGA REST server in Terminal 1
`./run.sh`

### Query the FPGA REST server from Terminal 2
`curl -d "url=https://upload.wikimedia.org/wikipedia/commons/d/de/Beagle_Upsy.jpg" -X POST http://localhost:5000/predict`

### Sample output
```
{                                                                                                                                                                         
  "predictions": [
    "0.3425 \"n02088364 beagle\"",
    "0.2108 \"n02089973 English foxhound\"",
    "0.2039 \"n02089867 Walker hound, Walker foxhound\"",
    "0.0400 \"n02101388 Brittany spaniel\"",
    "0.0359 \"n02099712 Labrador retriever\""
  ],
  "status": "success"
}
```
