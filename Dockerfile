FROM nvidia/cuda:10.2-runtime-ubuntu18.04
RUN apt-get update \
&& apt-get install -y software-properties-common \
&& add-apt-repository ppa:deadsnakes/ppa \
&& apt-get update \
&& apt-get install -y python3.7 \
        python3.7-dev \
        python3.7-distutils \
        curl \
        gcc \
        make \
       nano \
       curl
RUN  update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py
RUN pip install --upgrade pip \
                  setuptools
#RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh && chmod +x Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda && $HOME/miniconda/bin/conda init bash && /root/miniconda/bin/conda install cudatoolkit
