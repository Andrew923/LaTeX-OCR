FROM python:3.8-slim
RUN pip install torch>=1.7.1
WORKDIR /latexocr
ADD . /latexocr/
# ADD pix2tex /latexocr/pix2tex/
# ADD setup.py /latexocr/
# ADD README.md /latexocr/
RUN pip install -e .[api]
RUN python -m pix2tex.model.checkpoints.get_latest_checkpoint

ENV port = $PORT

# ENTRYPOINT ["uvicorn", "pix2tex.api.app:app", "--port", port]
CMD uvicorn pix2text.api.app:app --port 8080