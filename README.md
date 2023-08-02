# Sound_compressor

**This project is in process.**

It uses neural network to approximate wave and then uses Rice coding for loss. 

Current version compress the file a little bit, but training neural network is in progress.

# Installation

To install this sound compressor you have to install 2 libraries:

1. **libsndfile**
2. **torchscript**

To install **libsnfile** library run 

```
apt-get install libsndfile-dev
```

or you can copy this library from [GitHub](https://github.com/libsndfile/libsndfile).

To install **torchscript** follow the instructions from their [website](https://pytorch.org/cppdocs/installing.html), but instead of using their `CmakeLists.txt` file you should use  `CmakeLists.txt` from current repo.

# Usage 

After you installed all required libraries, building and testing sound_comressor is as simple as:

```
make
./compressor [option] [required file names or paths]
```

| Option | Required file names or paths| result |
|:------:|:---------------------------:|:------:|
|-h| doesn't require files | prints help information |
|-e| file to encode | encodes the input file (encoded file called *.nlac) |
|-d| file to decode and file to write decoded file | transforms the *.nlac file to *.wav
