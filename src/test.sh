#test locally

bentoml serve KerasFashionMnistService:latest --enable-microbatch
echo curl -X POST "http://127.0.0.1:5000/predict" -F image=@test_output_34_9.png