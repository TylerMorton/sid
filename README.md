# SID <img width="92" height="60" src="https://github.com/TylerMorton/sir/blob/main/docs/images/img-mQISWp9LB2E1tsfNOuPWmrzA.png">
SID (speech-to-image display) is a simple project to mess around with openAi's API for [speech-to-text](https://openai.com/research/whisper) and [text-to-image](https://openai.com/dall-e-2). My ultimate goal for the project is to have a physical setup of 1. a small monitor connected to 2. a raspberry pi. When a user clicks a button or some other activation they will be prompted to speak into a microphone and their query image will be displayed.
<div align="center">
  <div style="display: flex;">
  <img width="322" height="210" src="https://github.com/TylerMorton/sir/blob/main/docs/images/img-GeuyAsgvCaj5DMNA0ri9gdAP.png">
  <img width="322" height="210" src="https://github.com/TylerMorton/sir/blob/main/docs/images/img-Qpv502nStMATBT9UHeGzWaSG.png">
</div>
</div>


It's a good idea if you freshly installed rust to run these commands:
```sudo apt-get install build-essential pkg-config libasound2-dev llvm llvm-dev libclang-dev clang cmake

```
If having error `Permission denied` try removing recoded.wav from project directory.
