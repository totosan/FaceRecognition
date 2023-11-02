# Halloween AI with Face recognition

## Description

This projects intention is to make halloween a bit more interactive by recognizing kids, that already have visited a monster, that asked for the name.
With this I cover following challenges:

- detecting multiple faces
- fast and efficient
- tracking a face to keep  a temprary identity
- saving face images into a DB only if it is unknown
- recognizing a face with already detected faces
- getting an assignment between the name, the kid tells, and the image in the DB
- text to speech with the name from the recognized face
- it shouldn't lag
- it runs on a device w/o network
- device is Jetson Nano

## Installation
Because this projects targets Nvidias Jetson Nano you have to build OpenCV yourself with GPU and GStreamer support.
afterwards:

```bash
pip install -r requirements.txt
```

If several packages are not installing, try to wak through one by one. I put scikit and scipy wheels into the repo, so that you can install them directly (if errors here: the wheel may not target your Jetson Nano - you have to build it then)

From the root path of this repo clone SORT implementation from here:
```bash
git clone https://github.com/abewley/sort
```

## Contributing
If you want to contribute, feel free to contact me or simply create a PR