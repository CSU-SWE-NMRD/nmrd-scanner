# NMRD Pneumonia X-Ray Scanner

This is a scanner app that allows you to scan an X-ray of lungs for pneumonia.

## Description

The NMRD Pneumonia X-Ray Scanner app uses a computer vision model in order to classify X-rays as either pneumonia or normal. The model is pre-trained, but it demonstrates active learning through the report inaccuracy feature. Any X-Ray image can be scanned, and the program will output its classification and the confidence level. The statistics of the model can be viewed in the `statistics.json` file.

## Getting Started

### Dependencies

* Windows or Linux computer
* [uv](https://docs.astral.sh/uv/getting-started/installation/)
* [Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

### Installing

* Clone the repository
* Copy the `chest_xray` folder from the dataset into the `data` folder

### Executing program

* Open a terminal in the root directory of the project
* Run the command `uv run swe-project`

## Help

* Low confidence
  * Images that are very different from the dataset may cause the model to produce a low confidence score. Reporting the inaccuracy and retraining the model can help to improve this over time.

## Authors

Noah Faulk

Mason Thomas

RJ Norris

Darius Spurlin

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments

* [PyTorch Transfer Learning for Computer Vision Tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

