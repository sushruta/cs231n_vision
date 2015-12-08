from celery import Celery

app = Celery('tasks',
             backend='redis://localhost',
             broker='amqp://guest@localhost//')

@app.task
