# The complete explorer as a Hugging Face Space (SDK "docker"): the image holds the package `cypmode` and the folder `explorer/full` of the repository and nothing else (16 MB),
# installs the pinned packages of the explorer (including RDKit and xTB, which have no build for the browser), and runs the explorer as an application on port 7860.
FROM python:3.13-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates libgomp1 libxrender1 libxext6 libsm6 \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# The main branch of the repository at the time of the build; a factory rebuild of the Space takes the current one.
RUN git clone --quiet --depth 1 --branch main --filter=blob:none --sparse https://github.com/mohamedtalaat-gif/CYPMode.git . \
 && git sparse-checkout set cypmode explorer/full \
 && git log -1 --format="built from commit %h of %cd" --date=short
RUN pip install --no-cache-dir -r explorer/full/requirements.txt

EXPOSE 7860
CMD ["marimo", "run", "explorer/full/app.py", "--host", "0.0.0.0", "--port", "7860", "--headless", "--no-token"]
