### Prerequisites
1) [OpenCV 3.0](https://opencv.org/)
2) Download and install the [CUDA toolkit 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive)
for your corresponding platform. For system requirements and installation instructions of cuda toolkit, please refer to the [Windows Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).

### Description
This is CUDA console application that accepts a path to an image, and blur percent. With this input, it performes a two-pass guassian blur on the image on both the CPU and the GPU, then compares the time each method took. The GPU's performance time compared to the CPU's performance time decreases with image size and blur percent. Try it out for yourself.
![Mona_blur](https://github.com/user-attachments/assets/505b7f68-006b-4e63-8fff-35a9f7b4964a)
The displayed image is of of mona lisa blurred. The orginial image is the image 'Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg.webp' from the website 'https://en.wikipedia.org/wiki/Mona_Lisa'
