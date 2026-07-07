FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

EXPOSE 8080
ENV SENTINEL_HOST=0.0.0.0 SENTINEL_PORT=8080
# run_demo.py serves the dashboard with a live simulated model attached;
# the bare dashboard app would render empty until set_monitor() is called.
CMD ["python", "run_demo.py"]
