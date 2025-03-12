FROM public.ecr.aws/lambda/python:latest
# FROM python:3.11.7
# WORKDIR /api/
# COPY . .
RUN chmod -R 777 /var/task/

COPY . .

RUN python3 -m pip install -r requirements.txt

# WORKDIR /
# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD ["main.handler"]