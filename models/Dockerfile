# For ConNER 
# Woojae

FROM python:3.8

RUN apt-get update && apt-get install -y build-essential
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN git clone https://github.com/dmis-lab/ConNER.git
WORKDIR /ConNER
RUN pip install -r requirements.txt
RUN pip install notebook
CMD ["bash"]
