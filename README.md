# Fashion MNIST Deployment using BentoML
Implementation of a BentoML deployable model artifact using Fashion MNIST trained Keras Tensorflow model to AWS

## Prerequisites
1. AWS Account for deploying to AWS Sagemaker
2. Docker Client
3. VSCode (Optional)
4. Ubuntu/Mac (Optional - to run `.sh` scripts)

## Installation
1. Create and run in your own `conda` environment
2. Run `sh setup/install_requirements.sh`
3. Run `train_model.py`
4. Run `package_model.py`
5. Refer to `BentoML` docs for deploying to AWS Sagemaker
