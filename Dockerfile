#FROM python:3.7-alpine
FROM python:3.7-slim

WORKDIR /app
COPY requirements.txt ./requirements.txt
COPY . /app

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirements.txt
# RUN apt-get update
# RUN apt install -y libgl1-mesa-glx
# RUN apt update
# RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
#EXPOSE process.env.PORT || 8000
EXPOSE 8501
# ENTRYPOINT ["streamlit", "run"]
# CMD ["app.py"]

CMD streamlit run app.py
#CMD streamlit run --server.port $PORT app.py